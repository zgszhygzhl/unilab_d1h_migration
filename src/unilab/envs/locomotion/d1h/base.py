from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from unilab.base.backend import SimBackend
from unilab.base.np_env import NpEnvState
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common.base import (
    BaseNoiseConfig,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
)

D1H_JOINT_NAMES: tuple[str, ...] = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FL_foot_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FR_foot_joint",
)
D1H_SENSOR_PREFIXES: tuple[str, ...] = tuple(name.removesuffix("_joint") for name in D1H_JOINT_NAMES)
NUM_D1H_ACTIONS = 8
D1H_HIP_JOINT_INDICES = np.asarray([0, 4], dtype=np.int32)
D1H_FOOT_JOINT_INDICES = np.asarray([3, 7], dtype=np.int32)
D1H_LEG_JOINT_INDICES = np.asarray([0, 1, 2, 4, 5, 6], dtype=np.int32)


@dataclass
class D1HNoiseConfig(BaseNoiseConfig):
    pass


@dataclass
class D1HControlConfig:
    action_scale: float = 0.5
    hip_scale_reduction: float = 0.5
    clip_actions: float = 1.0
    simulate_action_latency: bool = False

    kp_hip: float = 40.0
    kd_hip: float = 1.0
    kp_thigh: float = 40.0
    kd_thigh: float = 1.0
    kp_calf: float = 40.0
    kd_calf: float = 1.0

    wheel_kp: float = 10.0
    wheel_kd: float = 0.5

    torque_limit_leg: float = 80.0
    torque_limit_wheel: float = 12.0


@dataclass
class D1HAsset:
    base_name: str = "base_link"
    foot_names: tuple[str, str] = ("FL_foot", "FR_foot")
    foot_joint_names: tuple[str, str] = ("FL_foot_joint", "FR_foot_joint")
    ground: str = "floor"


@dataclass
class D1HBaseCfg(LocomotionBaseCfg):
    noise_config: D1HNoiseConfig = field(default_factory=D1HNoiseConfig)  # type: ignore[assignment]
    control_config: D1HControlConfig = field(default_factory=D1HControlConfig)  # type: ignore[assignment]
    asset: D1HAsset = field(default_factory=D1HAsset)
    sim_dt: float = 0.0025
    ctrl_dt: float = 0.01


def stack_d1h_joint_sensors(backend: SimBackend, suffix: str, *, dtype: np.dtype | type) -> np.ndarray:
    names = tuple(f"{prefix}_{suffix}" for prefix in D1H_SENSOR_PREFIXES)
    values = backend.get_sensor_data_batch(names)
    return np.asarray(values.reshape(values.shape[0], -1)[:, :NUM_D1H_ACTIONS], dtype=dtype)


class D1HBaseEnv(LocomotionBaseEnv):
    _cfg: D1HBaseCfg

    hip_joint_indices = D1H_HIP_JOINT_INDICES
    foot_joint_indices = D1H_FOOT_JOINT_INDICES
    leg_joint_indices = D1H_LEG_JOINT_INDICES

    def __init__(self, cfg: D1HBaseCfg, backend: SimBackend, num_envs: int = 1):
        super().__init__(cfg, backend, num_envs)
        self._np_dtype = get_global_dtype()
        self._validate_d1h_contract(num_envs)
        self._kp = np.asarray(
            [
                cfg.control_config.kp_hip,
                cfg.control_config.kp_thigh,
                cfg.control_config.kp_calf,
                cfg.control_config.wheel_kp,
                cfg.control_config.kp_hip,
                cfg.control_config.kp_thigh,
                cfg.control_config.kp_calf,
                cfg.control_config.wheel_kp,
            ],
            dtype=self._np_dtype,
        )
        self._kd = np.asarray(
            [
                cfg.control_config.kd_hip,
                cfg.control_config.kd_thigh,
                cfg.control_config.kd_calf,
                cfg.control_config.wheel_kd,
                cfg.control_config.kd_hip,
                cfg.control_config.kd_thigh,
                cfg.control_config.kd_calf,
                cfg.control_config.wheel_kd,
            ],
            dtype=self._np_dtype,
        )

    def _init_action_space(self) -> None:
        self._action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(NUM_D1H_ACTIONS,),
            dtype=np.float32,
        )

    def _init_buffers(self) -> None:
        super()._init_buffers()
        if self.default_angles.shape != (NUM_D1H_ACTIONS,):
            raise ValueError(
                f"D1H home keyframe must provide {NUM_D1H_ACTIONS} joint qpos values, "
                f"got {self.default_angles.shape}"
            )

    def _validate_d1h_contract(self, num_envs: int) -> None:
        if self._backend.num_actuators != NUM_D1H_ACTIONS:
            raise ValueError(
                f"D1H requires {NUM_D1H_ACTIONS} motor actuators, got {self._backend.num_actuators}"
            )
        ctrl_range = np.asarray(self._backend.get_actuator_ctrl_range())
        if ctrl_range.shape != (NUM_D1H_ACTIONS, 2):
            raise ValueError(
                f"D1H actuator ctrl_range must have shape ({NUM_D1H_ACTIONS}, 2), got {ctrl_range.shape}"
            )
        expected_shape = (num_envs, NUM_D1H_ACTIONS)
        if self.get_dof_pos().shape != expected_shape:
            raise ValueError(f"D1H joint position sensors must have shape {expected_shape}")
        if self.get_dof_vel().shape != expected_shape:
            raise ValueError(f"D1H joint velocity sensors must have shape {expected_shape}")

    def get_dof_pos(self) -> np.ndarray:
        return stack_d1h_joint_sensors(self._backend, "pos", dtype=self.default_angles.dtype)

    def get_dof_vel(self) -> np.ndarray:
        return stack_d1h_joint_sensors(self._backend, "vel", dtype=self.default_angles.dtype)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        cfg = self._cfg.control_config
        clipped_actions = np.asarray(
            np.clip(actions, -float(cfg.clip_actions), float(cfg.clip_actions)),
            dtype=self._np_dtype,
        )
        if clipped_actions.shape != (self._num_envs, NUM_D1H_ACTIONS):
            raise ValueError(
                f"D1H actions must have shape ({self._num_envs}, {NUM_D1H_ACTIONS}), "
                f"got {clipped_actions.shape}"
            )

        state.info["last_actions"] = np.asarray(
            state.info.get("current_actions", np.zeros_like(clipped_actions)), dtype=self._np_dtype
        ).copy()
        state.info["current_actions"] = clipped_actions.copy()
        exec_actions = state.info["last_actions"] if cfg.simulate_action_latency else clipped_actions

        dof_pos = np.asarray(self.get_dof_pos(), dtype=self._np_dtype)
        dof_vel = np.asarray(self.get_dof_vel(), dtype=self._np_dtype)
        state.info["last_dof_pos"] = np.asarray(
            state.info.get("current_dof_pos", dof_pos), dtype=self._np_dtype
        ).copy()
        state.info["last_dof_vel"] = np.asarray(
            state.info.get("current_dof_vel", dof_vel), dtype=self._np_dtype
        ).copy()

        actions_scaled = exec_actions * float(cfg.action_scale)
        actions_scaled[:, D1H_HIP_JOINT_INDICES] *= float(cfg.hip_scale_reduction)
        targets = self.default_angles[None, :] + actions_scaled

        torques = np.zeros_like(clipped_actions, dtype=self._np_dtype)
        leg_ids = D1H_LEG_JOINT_INDICES
        torques[:, leg_ids] = self._kp[leg_ids] * (targets[:, leg_ids] - dof_pos[:, leg_ids])
        torques[:, leg_ids] -= self._kd[leg_ids] * dof_vel[:, leg_ids]
        wheel_ids = D1H_FOOT_JOINT_INDICES
        torques[:, wheel_ids] = self._kp[wheel_ids] * actions_scaled[:, wheel_ids]
        torques[:, wheel_ids] -= self._kd[wheel_ids] * dof_vel[:, wheel_ids]

        torques[:, leg_ids] = np.clip(
            torques[:, leg_ids],
            -float(cfg.torque_limit_leg),
            float(cfg.torque_limit_leg),
        )
        torques[:, wheel_ids] = np.clip(
            torques[:, wheel_ids],
            -float(cfg.torque_limit_wheel),
            float(cfg.torque_limit_wheel),
        )

        state.info["last_torques"] = np.asarray(
            state.info.get("current_torques", np.zeros_like(torques)), dtype=self._np_dtype
        ).copy()
        state.info["current_torques"] = torques.copy()
        state.info["torques"] = torques.copy()
        return torques