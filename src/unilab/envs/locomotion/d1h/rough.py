from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
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
from unilab.envs.locomotion.common.rewards import RewardContext, run_reward_dispatch
from unilab.envs.locomotion.d1h import rewards as d1h_rewards
from unilab.envs.locomotion.d1h.base import (
    NUM_D1H_ACTIONS,
    D1HBaseCfg,
    D1HBaseEnv,
)


@dataclass
class D1HCommands(Commands):
    vel_limit: list[list[float]] = field(default_factory=lambda: [[0.0, 0.0, -0.5], [0.5, 0.0, 0.5]])
    resampling_time: float = 6.0
    heading_command: bool = False
    rel_standing_envs: float = 0.1


@dataclass
class D1HTerminationConfig:
    min_base_height: float = 0.05
    max_tilt_deg: float = 50.0
    max_abs_xy: float | None = None


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
    desired_feet_distance: float = 0.38
    foot_contact_height: float = 0.11


@registry.envcfg("d1h_rough")
@dataclass
class D1HRoughCfg(D1HBaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "d1h" / "scene.xml")
        )
    )
    max_episode_seconds: float = 20.0
    commands: D1HCommands = field(default_factory=D1HCommands)
    termination_config: D1HTerminationConfig = field(default_factory=D1HTerminationConfig)
    reward_config: D1HRewardConfig | None = None

    num_actions: int = 8
    n_proprio: int = 36
    n_scan: int = 187
    history_len: int = 10
    n_priv_latent: int = 33
    policy_obs_dim: int = 616


@registry.env("d1h_rough", sim_backend="mujoco")
class D1HRoughEnv(D1HBaseEnv):
    _cfg: D1HRoughCfg

    def __init__(self, cfg: D1HRoughCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        if cfg.num_actions != NUM_D1H_ACTIONS:
            raise ValueError(f"D1H num_actions must be {NUM_D1H_ACTIONS}, got {cfg.num_actions}")
        if cfg.policy_obs_dim != cfg.n_proprio + cfg.n_scan + cfg.history_len * cfg.n_proprio + cfg.n_priv_latent:
            raise ValueError("D1H policy_obs_dim does not match proprio/scan/history/privileged dimensions")

        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            add_body_sensors=True,
            post_step_forward_sensor=cfg.post_step_forward_sensor,
        )
        super().__init__(cfg, backend, num_envs)
        self._backend.materialize()
        self._reward_cfg = cfg.reward_config
        self._enable_reward_log = True
        self._rng = np.random.default_rng()
        self._proprio_history = np.zeros((num_envs, cfg.history_len, cfg.n_proprio), dtype=get_global_dtype())
        self._last_dof_vel_for_acc = np.zeros((num_envs, NUM_D1H_ACTIONS), dtype=get_global_dtype())
        self._foot_body_ids = self._backend.get_body_ids(cfg.asset.foot_names)
        self._current_air_time = np.zeros((num_envs, len(cfg.asset.foot_names)), dtype=get_global_dtype())
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
        return {"obs": 616}

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        env_ids = np.asarray(env_indices, dtype=np.int32)
        num_reset = len(env_ids)
        qpos = np.tile(self._init_qpos, (num_reset, 1))
        qvel = np.tile(self._init_qvel, (num_reset, 1))
        self._backend.set_state(env_ids, qpos, qvel)

        commands = self._sample_commands(num_reset)
        info_updates = self._zero_info_updates(num_reset)
        info_updates["commands"] = commands
        if self._cfg.commands.heading_command:
            info_updates["heading_commands"] = sample_heading_commands(self, num_reset)

        linvel, gyro, up, dof_pos, dof_vel, base_height = self._collect_core_state(env_ids)
        proprio = self._compute_proprio(info_updates, linvel, gyro, up, dof_pos, dof_vel)
        self._proprio_history[env_ids, :, :] = proprio[:, None, :]
        self._last_dof_vel_for_acc[env_ids] = dof_vel
        self._current_air_time[env_ids] = 0.0
        self._current_contact_time[env_ids] = 0.0
        self._last_foot_contact[env_ids] = False
        self._update_foot_state(info_updates, env_ids=env_ids, reset=True)
        obs = self._assemble_obs(info_updates, proprio, linvel, up, dof_vel, base_height, env_ids=env_ids)
        return {"obs": obs}, info_updates

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_commands(state.info)
        linvel, gyro, up, dof_pos, dof_vel, base_height = self._collect_core_state()
        state.info["current_dof_pos"] = dof_pos.copy()
        state.info["current_dof_vel"] = dof_vel.copy()
        state.info["qacc"] = self._estimate_dof_acc(dof_vel)
        self._update_foot_state(state.info)
        proprio = self._compute_proprio(state.info, linvel, gyro, up, dof_pos, dof_vel)
        self._proprio_history = np.roll(self._proprio_history, shift=-1, axis=1)
        self._proprio_history[:, -1, :] = proprio
        terminated = self._compute_terminated(up, base_height, state.info)
        reward = self._compute_reward(state.info, linvel, gyro, up, dof_pos, dof_vel, base_height)
        obs = self._assemble_obs(state.info, proprio, linvel, up, dof_vel, base_height)
        return state.replace(obs={"obs": obs}, reward=reward, terminated=terminated)

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
                commands_arr[resample_mask] = self._sample_commands(int(np.count_nonzero(resample_mask)))
                if self._cfg.commands.heading_command:
                    heading_commands = self._ensure_heading_commands(info, commands_arr.shape[0])
                    heading_commands[resample_mask] = sample_heading_commands(self, int(np.count_nonzero(resample_mask)))
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

    def _collect_core_state(self, env_ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = None if env_ids is None else np.asarray(env_ids, dtype=np.intp)
        base_quat = np.asarray(self._backend.get_base_quat(), dtype=get_global_dtype())
        world_linvel = np.asarray(self._backend.get_base_lin_vel(), dtype=get_global_dtype())
        local_linvel = np_quat_apply_inverse(base_quat, world_linvel)
        gyro = np.asarray(self._backend.get_sensor_data("trunk_gyro"), dtype=get_global_dtype())
        up = np_quat_apply_inverse(
            base_quat,
            np.broadcast_to(np.asarray([0.0, 0.0, 1.0], dtype=get_global_dtype()), (self._num_envs, 3)),
        )
        dof_pos = np.asarray(self.get_dof_pos(), dtype=get_global_dtype())
        dof_vel = np.asarray(self.get_dof_vel(), dtype=get_global_dtype())
        base_height = np.asarray(self._backend.get_base_pos()[:, 2], dtype=get_global_dtype())
        if idx is not None:
            return local_linvel[idx], gyro[idx], up[idx], dof_pos[idx], dof_vel[idx], base_height[idx]
        return local_linvel, gyro, up, dof_pos, dof_vel, base_height

    def _compute_proprio(
        self,
        info: dict[str, Any],
        linvel: np.ndarray,
        gyro: np.ndarray,
        up: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> np.ndarray:
        num_obs = dof_pos.shape[0]
        noise_cfg = self._cfg.noise_config
        commands = np.asarray(info.get("commands", np.zeros((num_obs, 3))), dtype=get_global_dtype())
        actions = np.asarray(
            info.get("current_actions", np.zeros((num_obs, NUM_D1H_ACTIONS))),
            dtype=get_global_dtype(),
        )
        diff = dof_pos - self.default_angles[None, :]
        proprio = np.concatenate(
            [
                self._obs_noise(gyro, noise_cfg.scale_gyro),
                self._obs_noise(up, noise_cfg.scale_gravity),
                commands[:, :3],
                self._obs_noise(diff, noise_cfg.scale_joint_angle),
                self._obs_noise(dof_vel, noise_cfg.scale_joint_vel),
                actions,
                self._obs_noise(linvel, noise_cfg.scale_linvel),
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        if proprio.shape != (num_obs, self._cfg.n_proprio):
            raise ValueError(f"D1H proprio must have shape ({num_obs}, {self._cfg.n_proprio}), got {proprio.shape}")
        return proprio

    def _compute_height_scan(self, num_obs: int) -> np.ndarray:
        return np.zeros((num_obs, self._cfg.n_scan), dtype=get_global_dtype())

    def _compute_privileged_latent(
        self,
        info: dict[str, Any],
        linvel: np.ndarray,
        up: np.ndarray,
        dof_vel: np.ndarray,
        base_height: np.ndarray,
    ) -> np.ndarray:
        num_obs = linvel.shape[0]
        torques = np.asarray(info.get("current_torques", np.zeros((num_obs, NUM_D1H_ACTIONS))), dtype=get_global_dtype())
        actions = np.asarray(info.get("current_actions", np.zeros((num_obs, NUM_D1H_ACTIONS))), dtype=get_global_dtype())
        friction = np.ones((num_obs, 1), dtype=get_global_dtype())
        mass_placeholders = np.zeros((num_obs, 4), dtype=get_global_dtype())
        pieces = [
            linvel,
            base_height.reshape(num_obs, 1),
            up,
            friction,
            mass_placeholders,
            torques,
            actions,
            dof_vel[:, :5],
        ]
        latent = np.concatenate(pieces, axis=1, dtype=get_global_dtype())
        if latent.shape[1] < self._cfg.n_priv_latent:
            pad = np.zeros((num_obs, self._cfg.n_priv_latent - latent.shape[1]), dtype=get_global_dtype())
            latent = np.concatenate([latent, pad], axis=1, dtype=get_global_dtype())
        latent = latent[:, : self._cfg.n_priv_latent]
        if latent.shape != (num_obs, self._cfg.n_priv_latent):
            raise ValueError(f"D1H privileged latent must have shape ({num_obs}, {self._cfg.n_priv_latent}), got {latent.shape}")
        return latent

    def _assemble_obs(
        self,
        info: dict[str, Any],
        proprio: np.ndarray,
        linvel: np.ndarray,
        up: np.ndarray,
        dof_vel: np.ndarray,
        base_height: np.ndarray,
        *,
        env_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        num_obs = proprio.shape[0]
        history = self._proprio_history if env_ids is None else self._proprio_history[np.asarray(env_ids, dtype=np.intp)]
        obs = np.concatenate(
            [
                proprio,
                self._compute_height_scan(num_obs),
                history.reshape(num_obs, self._cfg.history_len * self._cfg.n_proprio),
                self._compute_privileged_latent(info, linvel, up, dof_vel, base_height),
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        if obs.shape != (num_obs, self._cfg.policy_obs_dim):
            raise ValueError(f"D1H obs must have shape ({num_obs}, {self._cfg.policy_obs_dim}), got {obs.shape}")
        return obs

    def _update_foot_state(self, info: dict[str, Any], *, env_ids: np.ndarray | None = None, reset: bool = False) -> None:
        idx = None if env_ids is None else np.asarray(env_ids, dtype=np.intp)
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        base_quat = np.asarray(self._backend.get_base_quat(), dtype=get_global_dtype())
        foot_pos_w = np.asarray(self._backend.get_body_pos_w(self._foot_body_ids), dtype=get_global_dtype())
        rel = foot_pos_w - base_pos[:, None, :]
        foot_pos_b = np_quat_apply_batched(np_quat_conjugate_batched(base_quat[:, None, :]), rel)
        contact = foot_pos_w[:, :, 2] <= float(self._reward_cfg.foot_contact_height)
        first_contact = contact & (~self._last_foot_contact)
        update_idx = np.arange(self._num_envs, dtype=np.intp) if idx is None else idx
        self._current_air_time[update_idx] = np.where(
            contact[update_idx],
            0.0,
            self._current_air_time[update_idx] + self._cfg.ctrl_dt,
        )
        self._current_contact_time[update_idx] = np.where(
            contact[update_idx],
            self._current_contact_time[update_idx] + self._cfg.ctrl_dt,
            0.0,
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
        info["stand_still_command_threshold"] = float(self._reward_cfg.stand_still_command_threshold)
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

    def _estimate_dof_acc(self, dof_vel: np.ndarray) -> np.ndarray:
        qacc = np.asarray((dof_vel - self._last_dof_vel_for_acc) / self._cfg.ctrl_dt, dtype=get_global_dtype())
        self._last_dof_vel_for_acc[:] = dof_vel
        return qacc