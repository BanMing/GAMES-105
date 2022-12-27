import numpy as np
from scipy.spatial.transform import Rotation as R

def get_world_position_rotation(
    joint_positions, joint_orientations, joint_parents, joint_index
):
    parent_index = 0
    for i in range(len(joint_parents)):
        if joint_index == joint_parents[i]:
            parent_index = i - 1
            break

    rotation = joint_orientations[joint_index]
    position = joint_positions[joint_index]
    for i in range(parent_index, -1, -1):
        parent_joint = joint_parents[parent_index]
        parent_quat = R.from_quat(joint_orientations[parent_joint])
        rotation = (R.from_quat(rotation) * parent_quat).as_quat()
        position = joint_positions[parent_joint] + parent_quat.apply(position)

    return position, rotation


def part1_inverse_kinematics(
    meta_data, joint_positions, joint_orientations, target_pose
):
    """
    完成函数，计算逆运动学
    输入:
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    effector_index = path1[0]
    joint_len = len(path)
    effector_position, _ = get_world_position_rotation(
        joint_positions, joint_orientations, path, effector_index
    )
    # for i in range(15):
    for j in range(joint_len - 2, -1, -1):
        cur_position, cur_rotation = get_world_position_rotation(
            joint_positions, joint_orientations, path, effector_index, path[j]
        )
        joint_to_target = target_pose - cur_position
        joint_to_effector = effector_position - cur_position
        test1 = R.from_rotvec(joint_to_target)
        test2 = R.from_rotvec(joint_to_effector)
        # test = R.reduce(test2, test1)
        # joint_orientations[path[j]] = (
        #     R.from_rotvec(joint_to_effector, joint_to_target)
        #     .as_quat()
        #     .apply(joint_orientations[path[j]])
        # )
        effector_position, _ = get_world_position_rotation(
            joint_positions, joint_orientations, path, effector_index
        )
    return joint_positions, joint_orientations


def part2_inverse_kinematics(
    meta_data,
    joint_positions,
    joint_orientations,
    relative_x,
    relative_z,
    target_height,
):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """

    return joint_positions, joint_orientations


def bonus_inverse_kinematics(
    meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose
):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """

    return joint_positions, joint_orientations
