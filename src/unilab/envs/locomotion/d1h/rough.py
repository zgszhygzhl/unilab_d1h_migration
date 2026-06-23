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
)


@dataclass
class D1HCommands(Commands):
    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0, -0.5], [0.5, 0.0, 0.5]]
    )
    resampling_time: float = 6.0
    heading_command: bool = False
    rel_standing_envs: float = 0.1


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
            "flat": flat(proportion=0.2),
            "pyramid_stairs": pyramid_stairs(
                proportion=0.8,
                step_height_range=(0.02, 0.16),
                step_width=0.4,
                platform_width=3.0,
                border_width=0.2,
            ),
        }
    )


@dataclass
class D1HRewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "tracking_lin_vel_x": 2.0,
            "tracking_lin_vel_y": 1.0,
            "tracking_ang_vel": 1.0,
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "base_height": -5.0,
            "orientation": -2.0,
            "action_rate": -0.01,
            "torque": -2.0e-5,
            "dof_vel": -1.0e-4,
            "alive": 0.2,
            "stand_still": -0.5,
            "feet_air_time": 0.2,
            "body_feet_distance_x": -0.5,
            "feet_distance": -0.5,
            "collision": -1.0,
        }
    )
    tracking_sigma: float = 0.25
    base_height_target: float = 0.16
    stand_still_command_threshold: float = 0.1
    desired_feet_distance: float = 0.44
    foot_contact_height: float = 0.11


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
    n_proprio: int = 36
    n_scan: int = 187
    history_len: int = 10
    n_priv_latent: int = 33
    critic_obs_dim: int = 616
    policy_obs_dim: int = 616


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
            cfg.n_proprio + cfg.n_scan + cfg.history_len * cfg.n_proprio + cfg.n_priv_latent
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
            "torque": d1h_rewards.torque,
            "dof_vel": d1h_rewards.dof_vel,
            "alive": d1h_rewards.alive,
            "stand_still": d1h_rewards.stand_still,
            "feet_air_time": d1h_rewards.feet_air_time,
            "body_feet_distance_x": d1h_rewards.body_feet_distance_x,
            "feet_distance": d1h_rewards.feet_distance,
            "collision": d1h_rewards.collision,
        }

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

        commands = self._sample_commands(num_reset)
        info_updates = self._zero_info_updates(num_reset)
        info_updates["commands"] = commands
        if self._cfg.commands.heading_command:
            info_updates["heading_commands"] = sample_heading_commands(self, num_reset)

        linvel, gyro, up, projected_gravity, dof_pos, dof_vel, base_height = self._collect_core_state(
            env_ids
        )
        actor_obs = self._compute_actor_obs(info_updates, gyro, projected_gravity, dof_pos, dof_vel)
        critic_proprio = self._compute_critic_proprio(
            linvel,
            gyro,
            projected_gravity,
            commands,
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
            projected_gravity,
            dof_vel,
            base_height,
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
        critic_proprio = self._compute_critic_proprio(
            linvel,
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
            projected_gravity,
            dof_vel,
            base_height,
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
        zeros_actions = np.zeros((num_envs, NUM_D1H_ACTIONS), dtype=get_global_dtype())
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
        }

    def _sample_commands(self, num_samples: int) -> np.ndarray:
        low = np.asarray(self._cfg.commands.vel_limit[0], dtype=get_global_dtype())
        high = np.asarray(self._cfg.commands.vel_limit[1], dtype=get_global_dtype())
        commands = sample_velocity_commands(self._rng, num_samples, low, high)
        standing_prob = float(getattr(self._cfg.commands, "rel_standing_envs", 0.0))
        if standing_prob > 0.0:
            standing = self._rng.uniform(size=(num_samples,)) < min(standing_prob, 1.0)
            commands[standing] = 0.0
        if self._cfg.commands.heading_command:
            commands[:, 2] = 0.0
        return np.asarray(commands, dtype=get_global_dtype())

    def _update_commands(self, info: dict[str, Any]) -> None:
        commands = info.get("commands")
        if commands is None:
            info["commands"] = self._sample_commands(self._num_envs)
            return

        commands_arr = np.asarray(commands, dtype=get_global_dtype())
        resampling_time = float(self._cfg.commands.resampling_time)
        if resampling_time > 0.0:
            interval_steps = max(int(round(resampling_time / self._cfg.ctrl_dt)), 1)
            steps = np.asarray(info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32)))
            resample_mask = (steps > 0) & ((steps % interval_steps) == 0)
            if np.any(resample_mask):
                num_resample = int(np.count_nonzero(resample_mask))
                commands_arr[resample_mask] = self._sample_commands(num_resample)
                if self._cfg.commands.heading_command:
                    heading_commands = self._ensure_heading_commands(info, commands_arr.shape[0])
                    heading_commands[resample_mask] = sample_heading_commands(self, num_resample)
                    info["heading_commands"] = heading_commands

        if self._cfg.commands.heading_command:
            heading_commands = self._ensure_heading_commands(info, commands_arr.shape[0])
            apply_heading_yaw_feedback(
                commands_arr,
                np.asarray(self._backend.get_base_quat(), dtype=get_global_dtype()),
                heading_commands,
                stiffness=float(self._cfg.commands.heading_control_stiffness),
                clip=1.0,
            )
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
        num_obs = gyro.shape[0]
        commands = np.asarray(info["commands"], dtype=get_global_dtype())
        actions = np.asarray(
            info.get("current_actions", np.zeros((num_obs, NUM_D1H_ACTIONS))),
            dtype=get_global_dtype(),
        )
        diff = dof_pos - self.default_angles[None, :]
        actor_obs = np.concatenate(
            [
                self._obs_noise(gyro, self._cfg.noise_config.scale_gyro),
                self._obs_noise(projected_gravity, self._cfg.noise_config.scale_gravity),
                commands[:, :3],
                self._obs_noise(diff, self._cfg.noise_config.scale_joint_angle),
                self._obs_noise(dof_vel, self._cfg.noise_config.scale_joint_vel),
                actions,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        if actor_obs.shape != (num_obs, self._cfg.actor_obs_dim):
            raise ValueError(
                f"D1H actor obs must have shape ({num_obs}, {self._cfg.actor_obs_dim}), "
                f"got {actor_obs.shape}"
            )
        return actor_obs

    def _compute_critic_proprio(
        self,
        linvel: np.ndarray,
        gyro: np.ndarray,
        projected_gravity: np.ndarray,
        commands: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        diff = dof_pos - self.default_angles[None, :]
        critic_proprio = np.concatenate(
            [
                linvel,
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
        if critic_proprio.shape != (linvel.shape[0], self._cfg.n_proprio):
            raise ValueError(
                f"D1H critic proprio must have shape ({linvel.shape[0]}, {self._cfg.n_proprio}), "
                f"got {critic_proprio.shape}"
            )
        return critic_proprio

    def _compute_critic_obs(
        self,
        critic_proprio: np.ndarray,
        info: dict[str, Any],
        linvel: np.ndarray,
        projected_gravity: np.ndarray,
        dof_vel: np.ndarray,
        base_height: np.ndarray,
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
        privileged = self._compute_privileged_latent(
            info=info,
            linvel=linvel,
            up=-projected_gravity,
            dof_vel=dof_vel,
            base_height=base_height,
        )
        critic_obs = np.concatenate(
            [
                critic_proprio,
                height_scan,
                history.reshape(num_obs, self._cfg.history_len * self._cfg.n_proprio),
                privileged,
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

    def _compute_privileged_latent(
        self,
        info: dict[str, Any],
        linvel: np.ndarray,
        up: np.ndarray,
        dof_vel: np.ndarray,
        base_height: np.ndarray,
    ) -> np.ndarray:
        num_obs = linvel.shape[0]
        torques = np.asarray(
            info.get("current_torques", np.zeros((num_obs, NUM_D1H_ACTIONS))),
            dtype=get_global_dtype(),
        )
        actions = np.asarray(
            info.get("current_actions", np.zeros((num_obs, NUM_D1H_ACTIONS))),
            dtype=get_global_dtype(),
        )
        friction = np.ones((num_obs, 1), dtype=get_global_dtype())
        mass_placeholders = np.zeros((num_obs, 4), dtype=get_global_dtype())
        latent = np.concatenate(
            [
                linvel,
                base_height.reshape(num_obs, 1),
                up,
                friction,
                mass_placeholders,
                torques,
                actions,
                dof_vel[:, :5],
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        if latent.shape[1] < self._cfg.n_priv_latent:
            pad = np.zeros(
                (num_obs, self._cfg.n_priv_latent - latent.shape[1]), dtype=get_global_dtype()
            )
            latent = np.concatenate([latent, pad], axis=1, dtype=get_global_dtype())
        latent = latent[:, : self._cfg.n_priv_latent]
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
        del base_height
        info["stand_still_command_threshold"] = float(
            self._reward_cfg.stand_still_command_threshold
        )
        info["desired_feet_distance"] = float(self._reward_cfg.desired_feet_distance)
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
            base_height=base_height_from_scan(self, linvel.shape[0]),
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