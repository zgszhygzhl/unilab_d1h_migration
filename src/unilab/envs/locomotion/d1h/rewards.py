from __future__ import annotations

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common.rewards import RewardContext


LEG_JOINT_INDICES = np.asarray([0, 1, 2, 4, 5, 6], dtype=np.intp)


def tracking_lin_vel_x(ctx: RewardContext) -> np.ndarray:
    commands = ctx.info["commands"]
    err = np.square(commands[:, 0] - ctx.linvel[:, 0])
    return np.asarray(np.exp(-err / ctx.tracking_sigma), dtype=get_global_dtype())


def tracking_lin_vel_y(ctx: RewardContext) -> np.ndarray:
    commands = ctx.info["commands"]
    err = np.square(commands[:, 1] - ctx.linvel[:, 1])
    return np.asarray(np.exp(-err / ctx.tracking_sigma), dtype=get_global_dtype())


def tracking_ang_vel(ctx: RewardContext) -> np.ndarray:
    commands = ctx.info["commands"]
    base_ang_vel = np.asarray(ctx.info.get("base_ang_vel", ctx.gyro), dtype=get_global_dtype())
    err = np.square(commands[:, 2] - base_ang_vel[:, 2])
    return np.asarray(np.exp(-err / ctx.tracking_sigma), dtype=get_global_dtype())


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


def powers(ctx: RewardContext) -> np.ndarray:
    assert ctx.dof_vel is not None
    torques = np.asarray(
        ctx.info.get("current_torques", np.zeros((ctx.num_envs, ctx.dof_pos.shape[1]))),
        dtype=get_global_dtype(),
    )
    return np.asarray(np.sum(np.abs(torques * ctx.dof_vel), axis=1), dtype=get_global_dtype())


def dof_vel(ctx: RewardContext) -> np.ndarray:
    assert ctx.dof_vel is not None
    return np.asarray(np.sum(np.square(ctx.dof_vel), axis=1), dtype=get_global_dtype())


def dof_acc(ctx: RewardContext) -> np.ndarray:
    qacc = np.asarray(
        ctx.info.get("qacc", np.zeros((ctx.num_envs, ctx.dof_pos.shape[1]))),
        dtype=get_global_dtype(),
    )
    return np.asarray(np.sum(np.square(qacc), axis=1), dtype=get_global_dtype())


def termination(ctx: RewardContext) -> np.ndarray:
    return np.asarray(
        ctx.info.get("termination", np.zeros((ctx.num_envs,), dtype=get_global_dtype())),
        dtype=get_global_dtype(),
    )


def stand_still(ctx: RewardContext) -> np.ndarray:
    threshold = float(ctx.info.get("stand_still_command_threshold", 0.1))
    stopped = np.linalg.norm(ctx.info["commands"][:, :2], axis=1) < threshold
    joint_pos_penalty = np.sum(
        np.abs(ctx.dof_pos[:, LEG_JOINT_INDICES] - ctx.default_angles[LEG_JOINT_INDICES]),
        axis=1,
    )
    lin_vel_penalty = np.sum(np.square(ctx.linvel[:, :2]), axis=1)
    return np.asarray((joint_pos_penalty + lin_vel_penalty) * stopped, dtype=get_global_dtype())


def feet_air_time(ctx: RewardContext) -> np.ndarray:
    air = np.asarray(ctx.info.get("current_air_time", np.zeros((ctx.num_envs, 2))), dtype=get_global_dtype())
    first_contact = np.asarray(ctx.info.get("feet_first_contact", np.zeros((ctx.num_envs, 2), dtype=bool)))
    reward = np.sum(air * first_contact, axis=1)
    lateral_sign = (np.abs(ctx.info["commands"][:, 1]) > 0.1).astype(get_global_dtype()) * 2.0 - 1.0
    return np.asarray(reward * lateral_sign, dtype=get_global_dtype())


def _foot_pos_b(ctx: RewardContext) -> np.ndarray:
    return np.asarray(ctx.info.get("foot_pos_b", np.zeros((ctx.num_envs, 2, 3))), dtype=get_global_dtype())


def body_pos_to_feet_x(ctx: RewardContext) -> np.ndarray:
    foot_pos_b = _foot_pos_b(ctx)
    distance_x = np.abs(np.mean(foot_pos_b[:, :, 0], axis=1))
    return np.asarray(np.exp(-distance_x / ctx.tracking_sigma), dtype=get_global_dtype())


def body_feet_distance_x(ctx: RewardContext) -> np.ndarray:
    foot_pos_b = _foot_pos_b(ctx)
    return np.asarray(np.square(np.abs(foot_pos_b[:, 0, 0] - foot_pos_b[:, 1, 0])), dtype=get_global_dtype())


def body_feet_distance_y(ctx: RewardContext) -> np.ndarray:
    foot_pos_b = _foot_pos_b(ctx)
    target = float(ctx.info.get("desired_feet_distance", 0.44))
    lateral = np.abs(foot_pos_b[:, 0, 1] - foot_pos_b[:, 1, 1])
    return np.asarray(np.square(np.abs(lateral - target)), dtype=get_global_dtype())


def feet_distance(ctx: RewardContext) -> np.ndarray:
    return body_feet_distance_y(ctx)


def body_symmetry_y(ctx: RewardContext) -> np.ndarray:
    foot_pos_b = _foot_pos_b(ctx)
    err = np.abs(np.abs(foot_pos_b[:, 0, 1]) - np.abs(foot_pos_b[:, 1, 1]))
    return np.asarray(np.exp(-err / ctx.tracking_sigma), dtype=get_global_dtype())


def body_symmetry_z(ctx: RewardContext) -> np.ndarray:
    foot_pos_b = _foot_pos_b(ctx)
    err = np.abs(np.abs(foot_pos_b[:, 0, 2]) - np.abs(foot_pos_b[:, 1, 2]))
    return np.asarray(np.exp(-err / ctx.tracking_sigma), dtype=get_global_dtype())


def no_gait(ctx: RewardContext) -> np.ndarray:
    contact = np.asarray(ctx.info.get("foot_contact", np.zeros((ctx.num_envs, 2), dtype=bool)))
    both_contact = np.sum(contact.astype(np.float32), axis=1) == 2
    return np.asarray(both_contact * (np.abs(ctx.info["commands"][:, 1]) < 0.1), dtype=get_global_dtype())


def both_feet_air(ctx: RewardContext) -> np.ndarray:
    contact = np.asarray(ctx.info.get("foot_contact", np.zeros((ctx.num_envs, 2), dtype=bool)))
    air = np.asarray(ctx.info.get("current_air_time", np.zeros((ctx.num_envs, 2))), dtype=get_global_dtype())
    both_air = ~np.any(contact, axis=1)
    grace = float(ctx.info.get("both_feet_air_grace_time", 0.04))
    ramp = max(float(ctx.info.get("both_feet_air_ramp_time", 0.08)), 1.0e-6)
    air_time = np.min(air, axis=1)
    return np.asarray(np.clip((air_time - grace) / ramp, 0.0, 1.0) * both_air, dtype=get_global_dtype())


def heading(ctx: RewardContext) -> np.ndarray:
    return np.zeros((ctx.num_envs,), dtype=get_global_dtype())


def upward(ctx: RewardContext) -> np.ndarray:
    if ctx.gravity is None:
        return np.zeros((ctx.num_envs,), dtype=get_global_dtype())
    return np.asarray(np.square(1.0 + ctx.gravity[:, 2]), dtype=get_global_dtype())


def head_los_distance(ctx: RewardContext) -> np.ndarray:
    foot_pos_b = _foot_pos_b(ctx)
    offset = float(ctx.info.get("head_los_forward_offset", 0.0))
    deadband = float(ctx.info.get("head_los_deadband", 0.05))
    max_distance = float(ctx.info.get("head_los_max_distance", 0.35))
    distance = np.abs(np.mean(foot_pos_b[:, :, 0], axis=1) + offset)
    distance = np.clip(distance - deadband, 0.0, max_distance)
    return np.asarray(np.square(distance), dtype=get_global_dtype())


def collision(ctx: RewardContext) -> np.ndarray:
    return np.asarray(
        ctx.info.get("collision", np.zeros((ctx.num_envs,), dtype=get_global_dtype())),
        dtype=get_global_dtype(),
    )
