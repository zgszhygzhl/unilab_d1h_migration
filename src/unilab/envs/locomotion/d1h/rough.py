from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg, TerrainSceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import (
    np_quat_apply_batched,
    np_quat_apply_inverse,
    np_quat_conjugate_batched,
)
from unilab.envs.locomotion.common.commands import (
    Commands,
    apply_heading_yaw_feedback,
    sample_heading_commands,
    sample_velocity_commands,
)
from unilab.envs.locomotion.common.height_scan import (
    HeightScanConfig,
    base_height_from_scan,
    height_scan_obs,
    init_height_scan_sensor,
    terrain_out_of_bounds,
)
from unilab.envs.locomotion.common.rewards import RewardContext, run_reward_dispatch
from unilab.envs.locomotion.common.terrain_spawn import (
    TerrainCurriculumCfg,
    TerrainSpawnManager,
)
from unilab.envs.locomotion.d1h import rewards as d1h_rewards
from unilab.envs.locomotion.d1h.base import NUM_D1H_ACTIONS, D1HBaseCfg, D1HBaseEnv
from unilab.terrains import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    flat,
    pyramid_stairs,
    pyramid_stairs_inv,
)


@dataclass
class D1HCommands(Commands):
    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-0.5, -0.2, -1.0], [0.5, 0.2, 1.0]]
    )
    commands_proportion: list[float] = field(
        default_factory=lambda: [0.45, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05]
    )
    resampling_time: float = 10.0
    heading_command: bool = False
    rel_standing_envs: float = 0.1
    max_lin_vel_x_change_rate: float = 0.5
    max_lin_vel_y_change_rate: float = 0.3
    max_ang_vel_change_rate: float = 0.5
    enable_command_buffer: bool = True
    buffer_smoothing_factor: float = 0.1
    flip_same_sign_probability: float = 0.2


@dataclass
class D1HTerminationConfig:
    min_base_height: float = 0.05
    max_tilt_deg: float = 50.0
    max_abs_xy: float | None = None
    terrain_out_of_bounds: bool = True
    terrain_distance_buffer: float = 3.0


@dataclass(kw_only=True)
class D1HRoughTerrainCfg(TerrainGeneratorCfg):
    size: tuple[float, float] = (8.0, 8.0)
    num_rows: int = 6
    num_cols: int = 6
    border_width: float = 1.0
    add_lights: bool = True
    horizontal_scale: float = 0.2

    sub_terrains: dict[str, SubTerrainCfg] = field(
        default_factory=lambda: {
            "flat": flat(proportion=0.1),
            "pyramid_stairs": pyramid_stairs(
                proportion=0.85,
                step_height_range=(0.015, 0.16),
                step_width=0.56,
                platform_width=3.0,
                border_width=0.2,
            ),
            "pyramid_stairs_inv": pyramid_stairs_inv(
                proportion=0.05,
                step_height_range=(0.015, 0.16),
                step_width=0.56,
                platform_width=3.0,
                border_width=0.2,
            ),
        }
    )



@dataclass
class D1HRewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "powers": -2.0e-5,
            "termination": -100.0,
            "tracking_lin_vel_x": 15.0,
            "tracking_lin_vel_y": 5.0,
            "tracking_ang_vel": 5.0,
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "dof_vel": 0.0,
            "dof_acc": -2.5e-7,
            "base_height": -23.0,
            "feet_air_time": 0.0,
            "collision": -10.0,
            "action_rate": -0.1,
            "stand_still": -1.0,
            "orientation": -10.0,
            "no_gait": 5.0,
            "both_feet_air": -10.0,
            "body_pos_to_feet_x": 1.0,
            "body_feet_distance_x": -50.0,
            "body_feet_distance_y": -100.0,
            "body_symmetry_y": 0.3,
            "body_symmetry_z": 0.9,
            "heading": 0.0,
            "upward": 1.0,
            "head_los_distance": -20.0,
        }
    )
    tracking_sigma: float = 0.25
    base_height_target: float = 0.5
    stand_still_command_threshold: float = 0.1
    desired_feet_distance: float = 0.44
    feet_distance_range: tuple[float, float] = (0.36, 0.50)
    foot_contact_height: float = 0.11
    head_los_forward_offset: float = 0.0
    head_los_deadband: float = 0.05
    head_los_max_distance: float = 0.35
    both_feet_air_contact_force: float = 1.0
    both_feet_air_grace_time: float = 0.04
    both_feet_air_ramp_time: float = 0.08


@registry.envcfg("d1h_rough")
@dataclass
class D1HRoughCfg(D1HBaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "d1h" / "robot.xml"),
            fragment_files=[
                str(ASSETS_ROOT_PATH / "robots" / "d1h" / "locomotion_task.xml"),
            ],
            terrain=TerrainSceneCfg(
                generator=D1HRoughTerrainCfg(),
                hfield_name="terrain_hfield",
                geom_name="floor",
            ),
        )
    )
    max_episode_seconds: float = 20.0
    commands: D1HCommands = field(default_factory=D1HCommands)
    terrain_scan: HeightScanConfig = field(default_factory=HeightScanConfig)
    terrain_curriculum: TerrainCurriculumCfg = field(default_factory=TerrainCurriculumCfg)
    termination_config: D1HTerminationConfig = field(default_factory=D1HTerminationConfig)
    reward_config: D1HRewardConfig | None = None

    num_actions: int = 8
    actor_obs_dim: int = 33
    n_proprio: int = 33
    n_scan: int = 187
    history_len: int = 10
    n_priv_latent: int = 36
    critic_obs_dim: int = 586
    policy_obs_dim: int = 586


@registry.env("d1h_rough", sim_backend="mujoco")
class D1HRoughEnv(D1HBaseEnv):
    _cfg: D1HRoughCfg

    def __init__(self, cfg: D1HRoughCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        if cfg.num_actions != NUM_D1H_ACTIONS:
            raise ValueError(f"D1H num_actions must be {NUM_D1H_ACTIONS}, got {cfg.num_actions}")
        if cfg.actor_obs_dim != 33:
            raise ValueError(f"D1H actor_obs_dim must be 33, got {cfg.actor_obs_dim}")
        expected_critic_dim = (
            cfg.n_proprio + cfg.n_scan + cfg.n_priv_latent + cfg.history_len * cfg.n_proprio
        )
        if cfg.critic_obs_dim != expected_critic_dim:
            raise ValueError(
                f"D1H critic_obs_dim must be {expected_critic_dim}, got {cfg.critic_obs_dim}"
            )
        if cfg.policy_obs_dim != cfg.critic_obs_dim:
            raise ValueError(
                f"D1H policy_obs_dim must equal critic_obs_dim {cfg.critic_obs_dim}, "
                f"got {cfg.policy_obs_dim}"
            )

        self._height_scan_dim = len(cfg.terrain_scan.measured_points_x) * len(
            cfg.terrain_scan.measured_points_y
        )
        if self._height_scan_dim != cfg.n_scan:
            raise ValueError(f"D1H height scan dim must be {cfg.n_scan}, got {self._height_scan_dim}")
        self._scene_terrain_origins: np.ndarray | None = None
        scene_cfg = cfg.scene
        terrain_generator = scene_cfg.terrain.generator if scene_cfg.terrain is not None else None
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            add_body_sensors=True,
            post_step_forward_sensor=cfg.post_step_forward_sensor,
        )
        terrain_origins = getattr(backend, "terrain_origins", None)
        if terrain_origins is not None:
            self._scene_terrain_origins = terrain_origins

        super().__init__(cfg, backend, num_envs)
        self._backend.materialize()
        if self._scene_terrain_origins is not None and terrain_generator is not None:
            self._spawn = TerrainSpawnManager(
                num_envs,
                self._scene_terrain_origins,
                cell_size=float(terrain_generator.size[0]),
                cfg=cfg.terrain_curriculum,
                terrain_surface_sampler=getattr(backend, "terrain_surface_sampler", None),
            )
        else:
            self._spawn = None
        init_height_scan_sensor(self, cfg.terrain_scan, cfg.asset.base_name)

        self._reward_cfg = cfg.reward_config
        self._enable_reward_log = True
        self._rng = np.random.default_rng()
        self._critic_proprio_history = np.zeros(
            (num_envs, cfg.history_len, cfg.n_proprio), dtype=get_global_dtype()
        )
        self._last_dof_vel_for_acc = np.zeros(
            (num_envs, NUM_D1H_ACTIONS), dtype=get_global_dtype()
        )
        self._foot_body_ids = self._backend.get_body_ids(cfg.asset.foot_names)
        self._current_air_time = np.zeros(
            (num_envs, len(cfg.asset.foot_names)), dtype=get_global_dtype()
        )
        self._current_contact_time = np.zeros_like(self._current_air_time)
        self._last_foot_contact = np.zeros((num_envs, len(cfg.asset.foot_names)), dtype=bool)
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel_x": d1h_rewards.tracking_lin_vel_x,
            "tracking_lin_vel_y": d1h_rewards.tracking_lin_vel_y,
            "tracking_ang_vel": d1h_rewards.tracking_ang_vel,
            "lin_vel_z": d1h_rewards.lin_vel_z,
            "ang_vel_xy": d1h_rewards.ang_vel_xy,
            "base_height": d1h_rewards.base_height,
            "orientation": d1h_rewards.orientation,
            "action_rate": d1h_rewards.action_rate,
            "powers": d1h_rewards.powers,
            "torque": d1h_rewards.torque,
            "dof_vel": d1h_rewards.dof_vel,
            "dof_acc": d1h_rewards.dof_acc,
            "termination": d1h_rewards.termination,
            "stand_still": d1h_rewards.stand_still,
            "feet_air_time": d1h_rewards.feet_air_time,
            "no_gait": d1h_rewards.no_gait,
            "both_feet_air": d1h_rewards.both_feet_air,
            "body_pos_to_feet_x": d1h_rewards.body_pos_to_feet_x,
            "body_feet_distance_x": d1h_rewards.body_feet_distance_x,
            "body_feet_distance_y": d1h_rewards.body_feet_distance_y,
            "feet_distance": d1h_rewards.body_feet_distance_y,
            "body_symmetry_y": d1h_rewards.body_symmetry_y,
            "body_symmetry_z": d1h_rewards.body_symmetry_z,
            "heading": d1h_rewards.heading,
            "upward": d1h_rewards.upward,
            "head_los_distance": d1h_rewards.head_los_distance,
            "collision": d1h_rewards.collision,
        }
        missing = {
            name
            for name, scale in self._reward_cfg.scales.items()
            if scale != 0 and name not in self._reward_fns
        }
        if missing:
            raise KeyError(f"D1H reward scales reference unregistered rewards: {sorted(missing)}")


    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": self._cfg.actor_obs_dim, "critic": self._cfg.critic_obs_dim}

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        env_ids = np.asarray(env_indices, dtype=np.int32)
        num_reset = len(env_ids)
        qpos = np.tile(self._init_qpos, (num_reset, 1))
        qvel = np.tile(self._init_qvel, (num_reset, 1))
        if self._spawn is not None:
            qpos[:, 0:2] += np.random.uniform(-0.5, 0.5, (num_reset, 2))
            qpos[:, 2] += 0.03
            qpos[:, 0:3] += self._spawn.origins_for(env_ids)
            self._spawn.record_episode_start(env_ids, qpos[:, 0:3])
        self._backend.set_state(env_ids, qpos, qvel)

        target_commands = self._sample_commands(num_reset)
        info_updates = self._zero_info_updates(num_reset)
        info_updates["target_commands"] = target_commands
        info_updates["commands"] = self._initial_smoothed_commands(target_commands)
        if self._cfg.commands.heading_command:
            info_updates["heading_commands"] = sample_heading_commands(self, num_reset)

        linvel, gyro, up, projected_gravity, dof_pos, dof_vel, base_height = self._collect_core_state(
            env_ids
        )
        actor_obs = self._compute_actor_obs(info_updates, gyro, projected_gravity, dof_pos, dof_vel)
        critic_proprio = self._compute_proprio(
            gyro,
            projected_gravity,
            info_updates["commands"],
            dof_pos,
            dof_vel,
            info_updates["current_actions"],
        )
        self._critic_proprio_history[env_ids, :, :] = critic_proprio[:, None, :]
        self._last_dof_vel_for_acc[env_ids] = dof_vel
        self._current_air_time[env_ids] = 0.0
        self._current_contact_time[env_ids] = 0.0
        self._last_foot_contact[env_ids] = False
        self._update_foot_state(info_updates, env_ids=env_ids, reset=True)
        critic_obs = self._compute_critic_obs(
            critic_proprio,
            info_updates,
            linvel,
            env_ids=env_ids,
        )
        return {"obs": actor_obs, "critic": critic_obs}, info_updates


    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_commands(state.info)
        linvel, gyro, up, projected_gravity, dof_pos, dof_vel, base_height = self._collect_core_state()
        state.info["current_dof_pos"] = dof_pos.copy()
        state.info["current_dof_vel"] = dof_vel.copy()
        state.info["qacc"] = self._estimate_dof_acc(dof_vel)
        self._update_foot_state(state.info)
        actions = np.asarray(
            state.info.get("current_actions", np.zeros((self._num_envs, NUM_D1H_ACTIONS))),
            dtype=get_global_dtype(),
        )
        actor_obs = self._compute_actor_obs(state.info, gyro, projected_gravity, dof_pos, dof_vel)
        critic_proprio = self._compute_proprio(
            gyro,
            projected_gravity,
            state.info["commands"],
            dof_pos,
            dof_vel,
            actions,
        )
        self._critic_proprio_history = np.roll(self._critic_proprio_history, shift=-1, axis=1)
        self._critic_proprio_history[:, -1, :] = critic_proprio
        terminated = self._compute_terminated(up, base_height, state.info)
        reward = self._compute_reward(state.info, linvel, gyro, up, dof_pos, dof_vel, base_height)
        critic_obs = self._compute_critic_obs(
            critic_proprio,
            state.info,
            linvel,
        )
        state = state.replace(
            obs={"obs": actor_obs, "critic": critic_obs}, reward=reward, terminated=terminated
        )
        done = state.terminated | state.truncated
        if self._spawn is not None and np.any(done):
            done_indices = np.where(done)[0]
            stats = self._spawn.update_on_done(
                done_indices,
                self._backend.get_base_pos()[done_indices],
            )
            if stats:
                state.info.setdefault("log", {})
                for key, value in stats.items():
                    state.info["log"][f"terrain_curriculum/{key}"] = float(value)
        return state


    def _zero_info_updates(self, num_envs: int) -> dict[str, np.ndarray]:
        dtype = get_global_dtype()
        zeros_actions = np.zeros((num_envs, NUM_D1H_ACTIONS), dtype=dtype)
        return {
            "current_actions": zeros_actions.copy(),
            "last_actions": zeros_actions.copy(),
            "current_torques": zeros_actions.copy(),
            "last_torques": zeros_actions.copy(),
            "torques": zeros_actions.copy(),
            "last_dof_pos": zeros_actions.copy(),
            "last_dof_vel": zeros_actions.copy(),
            "current_dof_pos": zeros_actions.copy(),
            "current_dof_vel": zeros_actions.copy(),
            "qacc": zeros_actions.copy(),
            "randomized_lag_tensor": np.zeros((num_envs, 1), dtype=dtype),
            "mass_params_tensor": np.zeros((num_envs, 4), dtype=dtype),
            "friction_coeffs_tensor": np.ones((num_envs, 1), dtype=dtype),
            "restitution_coeffs_tensor": np.zeros((num_envs, 1), dtype=dtype),
            "motor_strength": np.ones((num_envs, NUM_D1H_ACTIONS), dtype=dtype),
            "kp_factor": np.ones((num_envs, NUM_D1H_ACTIONS), dtype=dtype),
            "kd_factor": np.ones((num_envs, NUM_D1H_ACTIONS), dtype=dtype),
        }

    def _initial_smoothed_commands(self, target_commands: np.ndarray) -> np.ndarray:
        if bool(self._cfg.commands.enable_command_buffer):
            return np.zeros_like(target_commands, dtype=get_global_dtype())
        return np.asarray(target_commands, dtype=get_global_dtype()).copy()


    def _sample_commands(self, num_samples: int) -> np.ndarray:
        dtype = get_global_dtype()
        low = np.asarray(self._cfg.commands.vel_limit[0], dtype=dtype)
        high = np.asarray(self._cfg.commands.vel_limit[1], dtype=dtype)
        raw = self._rng.uniform(low=low, high=high, size=(num_samples, 3)).astype(dtype)
        commands = np.zeros((num_samples, 3), dtype=dtype)
        proportions = np.asarray(self._cfg.commands.commands_proportion, dtype=np.float64)
        if proportions.size != 9:
            raise ValueError("D1H commands_proportion must contain 9 mode probabilities")
        total = float(np.sum(proportions))
        if total <= 0.0:
            raise ValueError("D1H commands_proportion must have positive sum")
        modes = self._rng.choice(9, size=num_samples, p=proportions / total)
        commands[modes == 0, 0] = raw[modes == 0, 0]
        commands[modes == 1, 1] = raw[modes == 1, 1]
        mask = modes == 2
        commands[mask, :2] = raw[mask, :2]
        commands[modes == 3, 2] = raw[modes == 3, 2]
        mask = modes == 4
        commands[mask, 0] = raw[mask, 0]
        commands[mask, 2] = raw[mask, 2]
        mask = modes == 5
        commands[mask, 1] = raw[mask, 1]
        commands[mask, 2] = raw[mask, 2]
        mask = modes == 6
        commands[mask, :] = raw[mask, :]
        standing_prob = float(getattr(self._cfg.commands, "rel_standing_envs", 0.0))
        if standing_prob > 0.0:
            standing = self._rng.uniform(size=(num_samples,)) < min(standing_prob, 1.0)
            commands[standing] = 0.0
        if self._cfg.commands.heading_command:
            commands[:, 2] = 0.0
        return np.asarray(commands, dtype=dtype)

    def _maybe_flip_same_sign_commands(self, sampled: np.ndarray, previous: np.ndarray) -> np.ndarray:
        probability = float(getattr(self._cfg.commands, "flip_same_sign_probability", 0.0))
        if probability <= 0.0:
            return sampled
        out = sampled.copy()
        low = np.asarray(self._cfg.commands.vel_limit[0], dtype=get_global_dtype())
        high = np.asarray(self._cfg.commands.vel_limit[1], dtype=get_global_dtype())
        active = (np.abs(out) > 0.1) & (np.sign(out) == np.sign(previous))
        flip_mask = active & (self._rng.uniform(size=out.shape) < min(probability, 1.0))
        flipped = -out
        valid = (flipped >= low[None, :]) & (flipped <= high[None, :])
        out = np.where(flip_mask & valid, flipped, out)
        return out

    def _update_commands(self, info: dict[str, Any]) -> None:
        dtype = get_global_dtype()

        target = info.get("target_commands")
        commands = info.get("commands")

        # If commands are missing, initialize target_commands and smoothed commands.
        if target is None:
            target_arr = self._sample_commands(self._num_envs)
            info["target_commands"] = target_arr
            info["commands"] = self._initial_smoothed_commands(target_arr)
            return

        target_arr = np.asarray(target, dtype=dtype)
        commands_arr = np.asarray(
            commands if commands is not None else self._initial_smoothed_commands(target_arr),
            dtype=dtype,
        )

        # ------------------------------------------------------------
        # 1. Resample target commands at the configured interval.
        #    This corresponds to _resample_commands() in y1v0h_evt1_command.py.
        # ------------------------------------------------------------
        resampling_time = float(self._cfg.commands.resampling_time)
        if resampling_time > 0.0:
            interval_steps = max(int(round(resampling_time / self._cfg.ctrl_dt)), 1)
            steps = np.asarray(
                info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
            )

            resample_mask = (steps > 0) & ((steps % interval_steps) == 0)

            if np.any(resample_mask):
                # Keep the same probabilistic resampling logic as the original:
                # resample_prob = clamp(resampling_time / 10, 0.1, 1.0)
                resample_prob = np.clip(resampling_time / 10.0, 0.1, 1.0)
                candidate_ids = np.where(resample_mask)[0]
                keep = self._rng.uniform(size=(candidate_ids.shape[0],)) <= resample_prob
                resample_ids = candidate_ids[keep]

                if resample_ids.shape[0] > 0:
                    sampled = self._sample_commands(int(resample_ids.shape[0]))
                    target_arr[resample_ids] = sampled

                    if self._cfg.commands.heading_command:
                        heading_commands = self._ensure_heading_commands(info, target_arr.shape[0])
                        heading_commands[resample_ids] = sample_heading_commands(
                            self, int(resample_ids.shape[0])
                        )
                        info["heading_commands"] = heading_commands

        # ------------------------------------------------------------
        # 2. Heading command feedback.
        #    If heading_command=True, convert heading target to yaw-rate target.
        # ------------------------------------------------------------
        if self._cfg.commands.heading_command:
            heading_commands = self._ensure_heading_commands(info, target_arr.shape[0])
            apply_heading_yaw_feedback(
                target_arr,
                np.asarray(self._backend.get_base_quat(), dtype=dtype),
                heading_commands,
                stiffness=float(self._cfg.commands.heading_control_stiffness),
                clip=1.0,
            )

        # ------------------------------------------------------------
        # 3. Compute odometry velocity, matching original compute_given_commands().
        #
        # Original:
        #   odemetry_vel[:, :2] = base_lin_vel[:, :2]
        #   odemetry_vel[:, 2]  = base_ang_vel[:, 2]
        #
        # In UniLab:
        #   get_base_lin_vel() is world-frame, so rotate it into body frame.
        #   trunk_gyro is body-frame angular velocity.
        # ------------------------------------------------------------
        base_quat = np.asarray(self._backend.get_base_quat(), dtype=dtype)
        world_linvel = np.asarray(self._backend.get_base_lin_vel(), dtype=dtype)
        body_linvel = np_quat_apply_inverse(base_quat, world_linvel)

        try:
            body_angvel = np.asarray(self._backend.get_sensor_data("trunk_gyro"), dtype=dtype)
        except KeyError:
            body_angvel = np.asarray(self._backend.get_base_ang_vel(), dtype=dtype)

        odometry_vel = np.zeros_like(commands_arr, dtype=dtype)
        odometry_vel[:, :2] = body_linvel[:, :2]
        odometry_vel[:, 2] = body_angvel[:, 2]

        # ------------------------------------------------------------
        # 4. Compute smoothed commands, matching original logic.
        #
        # Original:
        #   command_diff = target_command - odometry_vel
        #   if abs(diff) > max_allowed_change:
        #       commands_given += sign(diff) * max_allowed_change
        #   else:
        #       commands_given = target_command
        #
        # Important:
        #   Do NOT multiply max_allowed_change by buffer_smoothing_factor.
        # ------------------------------------------------------------
        if bool(self._cfg.commands.enable_command_buffer):
            max_change_rates = np.asarray(
                [
                    self._cfg.commands.max_lin_vel_x_change_rate,
                    self._cfg.commands.max_lin_vel_y_change_rate,
                    self._cfg.commands.max_ang_vel_change_rate,
                ],
                dtype=dtype,
            )

            max_change_per_step = max_change_rates * float(self._cfg.ctrl_dt)

            command_diff = target_arr[:, :3] - odometry_vel[:, :3]
            diff_magnitude = np.abs(command_diff)

            for i in range(3):
                max_allowed_change = max_change_per_step[i]

                # Braking condition from original:
                # target command is close to zero and smaller than current odometry velocity.
                is_braking = (np.abs(target_arr[:, i]) < 0.1) & (
                    np.abs(target_arr[:, i]) <= np.abs(odometry_vel[:, i])
                )

                allowed_i = np.where(
                    is_braking,
                    2.0 * max_allowed_change,
                    max_allowed_change,
                ).astype(dtype)

                new_command_i = np.where(
                    diff_magnitude[:, i] > allowed_i,
                    commands_arr[:, i] + np.sign(command_diff[:, i]) * allowed_i,
                    target_arr[:, i],
                )

                commands_arr[:, i] = new_command_i
        else:
            commands_arr[:, :3] = target_arr[:, :3]

        info["target_commands"] = target_arr
        info["commands"] = commands_arr 


    def _ensure_heading_commands(self, info: dict[str, Any], num_obs: int) -> np.ndarray:
        heading_commands = info.get("heading_commands")
        if heading_commands is None or np.asarray(heading_commands).shape != (num_obs,):
            heading_commands = sample_heading_commands(self, num_obs)
        heading_commands = np.asarray(heading_commands, dtype=get_global_dtype())
        info["heading_commands"] = heading_commands
        return heading_commands

    def _collect_core_state(
        self, env_ids: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = None if env_ids is None else np.asarray(env_ids, dtype=np.intp)
        base_quat = np.asarray(self._backend.get_base_quat(), dtype=get_global_dtype())
        world_linvel = np.asarray(self._backend.get_base_lin_vel(), dtype=get_global_dtype())
        local_linvel = np_quat_apply_inverse(base_quat, world_linvel)
        gyro = np.asarray(self._backend.get_sensor_data("trunk_gyro"), dtype=get_global_dtype())
        up = np_quat_apply_inverse(
            base_quat,
            np.broadcast_to(
                np.asarray([0.0, 0.0, 1.0], dtype=get_global_dtype()), (self._num_envs, 3)
            ),
        )
        projected_gravity = -up
        dof_pos = np.asarray(self.get_dof_pos(), dtype=get_global_dtype())
        dof_vel = np.asarray(self.get_dof_vel(), dtype=get_global_dtype())
        base_height = base_height_from_scan(self, self._num_envs)
        if idx is not None:
            return (
                local_linvel[idx],
                gyro[idx],
                up[idx],
                projected_gravity[idx],
                dof_pos[idx],
                dof_vel[idx],
                base_height[idx],
            )
        return local_linvel, gyro, up, projected_gravity, dof_pos, dof_vel, base_height

    def _compute_actor_obs(
        self,
        info: dict[str, Any],
        gyro: np.ndarray,
        projected_gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> np.ndarray:
        actions = np.asarray(
            info.get("current_actions", np.zeros((gyro.shape[0], NUM_D1H_ACTIONS))),
            dtype=get_global_dtype(),
        )
        return self._compute_proprio(
            gyro,
            projected_gravity,
            np.asarray(info["commands"], dtype=get_global_dtype()),
            dof_pos,
            dof_vel,
            actions,
        )

    def _compute_proprio(
        self,
        gyro: np.ndarray,
        projected_gravity: np.ndarray,
        commands: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        num_obs = gyro.shape[0]
        diff = np.asarray(dof_pos - self.default_angles[None, :], dtype=get_global_dtype()).copy()
        diff[:, [3, 7]] = 0.0
        proprio = np.concatenate(
            [
                gyro,
                projected_gravity,
                commands[:, :3],
                diff,
                dof_vel,
                actions,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        if proprio.shape != (num_obs, self._cfg.n_proprio):
            raise ValueError(
                f"D1H proprio must have shape ({num_obs}, {self._cfg.n_proprio}), "
                f"got {proprio.shape}"
            )
        return proprio

    def _compute_critic_obs(
        self,
        critic_proprio: np.ndarray,
        info: dict[str, Any],
        linvel: np.ndarray,
        *,
        env_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        num_obs = critic_proprio.shape[0]
        if env_ids is None:
            height_scan = height_scan_obs(self, self._cfg.terrain_scan, num_obs)
            history = self._critic_proprio_history
        else:
            ids = np.asarray(env_ids, dtype=np.intp)
            height_scan = height_scan_obs(self, self._cfg.terrain_scan, self._num_envs)[ids]
            history = self._critic_proprio_history[ids]
        privileged = self._compute_privileged_latent(info=info, linvel=linvel)
        critic_obs = np.concatenate(
            [
                critic_proprio,
                height_scan,
                privileged,
                history.reshape(num_obs, self._cfg.history_len * self._cfg.n_proprio),
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        if critic_obs.shape != (num_obs, self._cfg.critic_obs_dim):
            raise ValueError(
                f"D1H critic obs must have shape ({num_obs}, {self._cfg.critic_obs_dim}), "
                f"got {critic_obs.shape}"
            )
        return critic_obs

    def _info_array(self, info: dict[str, Any], name: str, shape: tuple[int, int], fill: float) -> np.ndarray:
        value = info.get(name)
        if value is None:
            return np.full(shape, fill, dtype=get_global_dtype())
        arr = np.asarray(value, dtype=get_global_dtype())
        if arr.shape != shape:
            return np.resize(arr, shape).astype(get_global_dtype(), copy=False)
        return arr

    def _compute_privileged_latent(
        self,
        info: dict[str, Any],
        linvel: np.ndarray,
    ) -> np.ndarray:
        num_obs = linvel.shape[0]
        foot_contact = np.asarray(
            info.get("foot_contact", np.zeros((num_obs, 2), dtype=bool)), dtype=get_global_dtype()
        )
        latent = np.concatenate(
            [
                linvel,
                foot_contact - 0.5,
                self._info_array(info, "randomized_lag_tensor", (num_obs, 1), 0.0),
                self._info_array(info, "mass_params_tensor", (num_obs, 4), 0.0),
                self._info_array(info, "friction_coeffs_tensor", (num_obs, 1), 1.0),
                self._info_array(info, "restitution_coeffs_tensor", (num_obs, 1), 0.0),
                self._info_array(info, "motor_strength", (num_obs, NUM_D1H_ACTIONS), 1.0),
                self._info_array(info, "kp_factor", (num_obs, NUM_D1H_ACTIONS), 1.0),
                self._info_array(info, "kd_factor", (num_obs, NUM_D1H_ACTIONS), 1.0),
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        if latent.shape != (num_obs, self._cfg.n_priv_latent):
            raise ValueError(
                f"D1H privileged latent must have shape ({num_obs}, {self._cfg.n_priv_latent}), "
                f"got {latent.shape}"
            )
        return latent


    def _update_foot_state(
        self, info: dict[str, Any], *, env_ids: np.ndarray | None = None, reset: bool = False
    ) -> None:
        idx = None if env_ids is None else np.asarray(env_ids, dtype=np.intp)
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        base_quat = np.asarray(self._backend.get_base_quat(), dtype=get_global_dtype())
        foot_pos_w = np.asarray(self._backend.get_body_pos_w(self._foot_body_ids), dtype=get_global_dtype())
        rel = foot_pos_w - base_pos[:, None, :]
        foot_pos_b = np_quat_apply_batched(np_quat_conjugate_batched(base_quat[:, None, :]), rel)
        contact = self._compute_foot_contact(foot_pos_w)
        first_contact = contact & (~self._last_foot_contact)
        update_idx = np.arange(self._num_envs, dtype=np.intp) if idx is None else idx
        self._current_air_time[update_idx] = np.where(
            contact[update_idx], 0.0, self._current_air_time[update_idx] + self._cfg.ctrl_dt
        )
        self._current_contact_time[update_idx] = np.where(
            contact[update_idx], self._current_contact_time[update_idx] + self._cfg.ctrl_dt, 0.0
        )
        if reset and idx is not None:
            self._current_air_time[idx] = 0.0
            self._current_contact_time[idx] = 0.0
            first_contact[idx] = False
        self._last_foot_contact[update_idx] = contact[update_idx]

        if idx is None:
            info["foot_pos_b"] = foot_pos_b
            info["foot_contact"] = contact
            info["feet_first_contact"] = first_contact
            info["current_air_time"] = self._current_air_time.copy()
            info["current_contact_time"] = self._current_contact_time.copy()
        else:
            info["foot_pos_b"] = foot_pos_b[idx]
            info["foot_contact"] = contact[idx]
            info["feet_first_contact"] = first_contact[idx]
            info["current_air_time"] = self._current_air_time[idx].copy()
            info["current_contact_time"] = self._current_contact_time[idx].copy()

    def _compute_foot_contact(self, foot_pos_w: np.ndarray) -> np.ndarray:
        names = ("FL_foot_contact", "FR_foot_contact")
        try:
            contacts = [self._backend.get_sensor_data(name).reshape(self._num_envs, -1)[:, 0] for name in names]
        except KeyError:
            return foot_pos_w[:, :, 2] <= float(self._reward_cfg.foot_contact_height)
        return np.asarray(np.stack(contacts, axis=1) > 0.1, dtype=bool)

    def _compute_terminated(self, up: np.ndarray, base_height: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        min_base_height = float(self._cfg.termination_config.min_base_height)
        max_tilt_cos = float(np.cos(np.deg2rad(self._cfg.termination_config.max_tilt_deg)))
        terminated = (base_height < min_base_height) | (up[:, 2] < max_tilt_cos)
        max_abs_xy = self._cfg.termination_config.max_abs_xy
        if max_abs_xy is not None:
            base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
            terminated |= np.any(np.abs(base_pos[:, :2]) > float(max_abs_xy), axis=1)
        info["collision"] = terminated.astype(get_global_dtype())
        info["termination"] = terminated.astype(get_global_dtype())
        return np.asarray(terminated, dtype=bool)

    def _compute_reward(
        self,
        info: dict[str, Any],
        linvel: np.ndarray,
        gyro: np.ndarray,
        up: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        base_height: np.ndarray,
    ) -> np.ndarray:
        info["stand_still_command_threshold"] = float(
            self._reward_cfg.stand_still_command_threshold
        )
        info["desired_feet_distance"] = float(self._reward_cfg.desired_feet_distance)
        info["feet_distance_range"] = np.asarray(
            self._reward_cfg.feet_distance_range, dtype=get_global_dtype()
        )
        info["head_los_forward_offset"] = float(self._reward_cfg.head_los_forward_offset)
        info["head_los_deadband"] = float(self._reward_cfg.head_los_deadband)
        info["head_los_max_distance"] = float(self._reward_cfg.head_los_max_distance)
        info["both_feet_air_contact_force"] = float(self._reward_cfg.both_feet_air_contact_force)
        info["both_feet_air_grace_time"] = float(self._reward_cfg.both_feet_air_grace_time)
        info["both_feet_air_ramp_time"] = float(self._reward_cfg.both_feet_air_ramp_time)
        info["base_ang_vel"] = np.asarray(self._backend.get_base_ang_vel(), dtype=get_global_dtype())
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            num_envs=linvel.shape[0],
            default_angles=self.default_angles.astype(get_global_dtype()),
            tracking_sigma=float(self._reward_cfg.tracking_sigma),
            base_height_target=float(self._reward_cfg.base_height_target),
            base_height=base_height,
            gravity=up,
        )
        return run_reward_dispatch(
            scales=self._reward_cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
        )


    def _compute_truncated(self, state: NpEnvState) -> np.ndarray:
        truncated = super()._compute_truncated(state)
        if self._cfg.termination_config.terrain_out_of_bounds:
            terrain_scene = self._cfg.scene.terrain
            terrain_cfg = terrain_scene.generator if terrain_scene is not None else None
            np.logical_or(
                truncated,
                terrain_out_of_bounds(
                    self,
                    terrain_cfg,
                    float(self._cfg.termination_config.terrain_distance_buffer),
                ),
                out=truncated,
            )
        return truncated

    def _estimate_dof_acc(self, dof_vel: np.ndarray) -> np.ndarray:
        qacc = np.asarray(
            (dof_vel - self._last_dof_vel_for_acc) / self._cfg.ctrl_dt,
            dtype=get_global_dtype(),
        )
        self._last_dof_vel_for_acc[:] = dof_vel
        return qacc