from __future__ import annotations

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common.rewards import RewardContext


def _upright_gate(ctx: RewardContext) -> np.ndarray:
    if ctx.gravity is None:
        return np.ones((ctx.num_envs,), dtype=get_global_dtype())
    return np.asarray(np.clip(ctx.gravity[:, 2], 0.0, 1.0), dtype=get_global_dtype())


def _tracking_sigma(ctx: RewardContext, command: np.ndarray) -> np.ndarray:
    sigma = float(ctx.tracking_sigma)
    return sigma * (0.1 + np.abs(command)) / (0.25 + np.abs(command) + 1.0e-8)


def tracking_lin_vel_x(ctx: RewardContext) -> np.ndarray:
    commands = ctx.info["commands"]
    err = np.clip(np.square(commands[:, 0] - ctx.linvel[:, 0]), 0.0, 1.0)
    return np.asarray(_upright_gate(ctx) * np.exp(-err / _tracking_sigma(ctx, commands[:, 0])), dtype=get_global_dtype())


def tracking_lin_vel_y(ctx: RewardContext) -> np.ndarray:
    commands = ctx.info["commands"]
    err = np.clip(np.square(commands[:, 1] - ctx.linvel[:, 1]), 0.0, 1.0)
    return np.asarray(_upright_gate(ctx) * np.exp(-err / _tracking_sigma(ctx, commands[:, 1])), dtype=get_global_dtype())


def tracking_ang_vel(ctx: RewardContext) -> np.ndarray:
    commands = ctx.info["commands"]
    err = np.square(commands[:, 2] - ctx.gyro[:, 2])
    return np.asarray(_upright_gate(ctx) * np.exp(-err / _tracking_sigma(ctx, commands[:, 2])), dtype=get_global_dtype())


def lin_vel_z(ctx: RewardContext) -> np.ndarray:
    return np.asarray(np.square(ctx.linvel[:, 2]), dtype=get_global_dtype())


def ang_vel_xy(ctx: RewardContext) -> np.ndarray:
    return np.asarray(np.sum(np.square(ctx.gyro[:, :2]), axis=1), dtype=get_global_dtype())


def base_height(ctx: RewardContext) -> np.ndarray:
    return np.asarray(np.square(ctx.base_height - ctx.base_height_target), dtype=get_global_dtype())


def orientation(ctx: RewardContext) -> np.ndarray:
    if ctx.gravity is None:
        return np.zeros((ctx.num_envs,), dtype=get_global_dtype())
    return np.asarray(np.sum(np.square(ctx.gravity[:, :2]), axis=1), dtype=get_global_dtype())


def action_rate(ctx: RewardContext) -> np.ndarray:
    return np.asarray(
        np.sum(np.square(ctx.info["current_actions"] - ctx.info["last_actions"]), axis=1),
        dtype=get_global_dtype(),
    )


def torque(ctx: RewardContext) -> np.ndarray:
    torques = np.asarray(
        ctx.info.get("current_torques", np.zeros((ctx.num_envs, ctx.dof_pos.shape[1]))),
        dtype=get_global_dtype(),
    )
    return np.asarray(np.sum(np.square(torques), axis=1), dtype=get_global_dtype())


def dof_vel(ctx: RewardContext) -> np.ndarray:
    assert ctx.dof_vel is not None
    return np.asarray(np.sum(np.square(ctx.dof_vel), axis=1), dtype=get_global_dtype())


def alive(ctx: RewardContext) -> np.ndarray:
    return np.ones((ctx.num_envs,), dtype=get_global_dtype())


def stand_still(ctx: RewardContext) -> np.ndarray:
    threshold = float(ctx.info.get("stand_still_command_threshold", 0.1))
    stopped = np.linalg.norm(ctx.info["commands"][:, :2], axis=1) < threshold
    err = np.sum(np.abs(ctx.dof_pos - ctx.default_angles), axis=1)
    return np.asarray(err * stopped, dtype=get_global_dtype())


def feet_air_time(ctx: RewardContext) -> np.ndarray:
    air = np.asarray(ctx.info.get("current_air_time", np.zeros((ctx.num_envs, 2))), dtype=get_global_dtype())
    first_contact = np.asarray(ctx.info.get("feet_first_contact", np.zeros((ctx.num_envs, 2), dtype=bool)))
    moving = np.linalg.norm(ctx.info["commands"][:, :2], axis=1) > 0.1
    reward = np.sum((air - 0.25) * first_contact, axis=1)
    return np.asarray(np.maximum(reward, 0.0) * moving, dtype=get_global_dtype())


def body_feet_distance_x(ctx: RewardContext) -> np.ndarray:
    foot_pos_b = np.asarray(ctx.info.get("foot_pos_b", np.zeros((ctx.num_envs, 2, 3))), dtype=get_global_dtype())
    return np.asarray(np.square(foot_pos_b[:, 0, 0] - foot_pos_b[:, 1, 0]), dtype=get_global_dtype())


def feet_distance(ctx: RewardContext) -> np.ndarray:
    foot_pos_b = np.asarray(ctx.info.get("foot_pos_b", np.zeros((ctx.num_envs, 2, 3))), dtype=get_global_dtype())
    target = float(ctx.info.get("desired_feet_distance", 0.38))
    lateral = np.abs(foot_pos_b[:, 0, 1] - foot_pos_b[:, 1, 1])
    return np.asarray(np.square(lateral - target), dtype=get_global_dtype())


def collision(ctx: RewardContext) -> np.ndarray:
    return np.asarray(ctx.info.get("collision", np.zeros((ctx.num_envs,), dtype=get_global_dtype())), dtype=get_global_dtype())