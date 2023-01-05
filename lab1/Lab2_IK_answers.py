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
        parent_joint = joint_parents[i]
        parent_quat = R.from_quat(joint_orientations[parent_joint])
        rotation = (R.from_quat(rotation) * parent_quat).as_quat()
        position = joint_positions[parent_joint] + parent_quat.apply(position)

    return position, rotation


def get_chain_list(meta_data, joint_positions, joint_orientations):
    path, _, path1, path2 = meta_data.get_path_from_root_to_end()
    chain_positions = []
    chain_orientations = []
    chain_offsets = []
    chain_offsets.append(np.array([0.0, 0.0, 0.0]))  # 根节点
    for i in range(len(path)):
        chain_positions.append(joint_positions[path[i]])
        chain_orientations.append(R.from_quat(joint_orientations[path[i]]))
        if i + 1 < len(path):
            chain_offsets.append(
                meta_data.joint_initial_position[path[i + 1]]
                - meta_data.joint_initial_position[path[i]]
            )

    return chain_positions, chain_orientations, chain_offsets


def ccd(chain_positions, chain_orientations, chain_offsets, target_pose):
    effector_index = len(chain_positions) - 1
    chain_len = len(chain_positions)
    count = 0
    while (
        np.linalg.norm(chain_positions[effector_index] - target_pose) >= 1e-2
        and count <= 10
    ):
        for i in range(chain_len - 2, -1, -1):
            effector_position = chain_positions[effector_index]
            cur_position = chain_positions[i]

            joint_to_target = target_pose - cur_position
            joint_to_effector = effector_position - cur_position

            # 计算两向量间旋转：https://www.xarg.org/proof/ quaternion-from-two-vectors/
            w = np.cross(joint_to_effector, joint_to_target)
            d = np.dot(joint_to_effector, joint_to_target)
            rotation = R.from_quat(
                [w[0], w[1], w[2], d + np.sqrt(d * d + np.dot(w, w))]
            )

            chain_orientations[i] = rotation * chain_orientations[i]

            local_rotations = []
            local_rotations.append(chain_orientations[0])
            for j in range(chain_len - 1):
                local_rotations.append(
                    R.inv(chain_orientations[j]) * chain_orientations[j + 1]
                )

            # 更新当前节点到尾节点的数据
            for j in range(i, effector_index):
                chain_positions[j + 1] = chain_positions[j] + chain_orientations[
                    j
                ].apply(chain_offsets[j + 1])
                if j + 1 < effector_index:
                    chain_orientations[j + 1] = (
                        chain_orientations[j] * local_rotations[j + 1]
                    )
                else:
                    chain_orientations[j + 1] = chain_orientations[j]
        count += 1


def apply_ik(
    meta_data, joint_positions, joint_orientations, chain_positions, chain_orientations
):
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()

    # 获得关节本地旋转，需要先求
    local_rotations = R.identity(len(meta_data.joint_name))
    for i in range(len(meta_data.joint_parent)):
        if meta_data.joint_parent[i] == -1:
            local_rotations[i] = R.from_quat(joint_orientations[i])
        else:
            local_rotations[i] = R.inv(
                R.from_quat(joint_orientations[meta_data.joint_parent[i]])
            ) * R.from_quat(joint_orientations[i])

    # 两条链路的情况
    if len(path2) > 1:
        # 更新链路节点到骨骼中
        for i in range(len(path2) - 1):
            joint_orientations[path2[i + 1]] = chain_orientations[i].as_quat()
        joint_orientations[path2[-1]] = chain_orientations[len(path2) - 1].as_quat()

        # path1返回从根节点到手的路径,所以需要逆向
        for i in range(len(path1) - 1):
            joint_orientations[path1[~i]] = chain_orientations[i + len(path2)].as_quat()

    for i in range(len(path)):
        joint_positions[path[i]] = chain_positions[i]
        if len(path2) < 2:
            joint_orientations[path[i]] = chain_orientations[i].as_quat()

    # 跟新其余节点
    for i in range(len(meta_data.joint_parent)):
        if meta_data.joint_parent[i] == -1:
            continue
        if meta_data.joint_name[i] not in path_name:
            joint_positions[i] = joint_positions[
                meta_data.joint_parent[i]
            ] + R.from_quat(joint_orientations[meta_data.joint_parent[i]]).apply(
                meta_data.joint_initial_position[i]
                - meta_data.joint_initial_position[meta_data.joint_parent[i]]
            )
            joint_orientations[i] = (
                R.from_quat(joint_orientations[meta_data.joint_parent[i]])
                * local_rotations[i]
            ).as_quat()


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
    # 值得注意的是传入的都是世界坐标系下的数据，并不是本地坐标和本地旋转

    chain_positions, chain_orientations, chain_offsets = get_chain_list(
        meta_data, joint_positions, joint_orientations
    )

    ccd(chain_positions, chain_orientations, chain_offsets, target_pose)

    apply_ik(
        meta_data,
        joint_positions,
        joint_orientations,
        chain_positions,
        chain_orientations,
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

    target_pose = np.array(
        [
            relative_x + joint_positions[0][0],
            target_height,
            relative_z + joint_positions[0][2],
        ]
    )

    chain_positions, chain_orientations, chain_offsets = get_chain_list(
        meta_data, joint_positions, joint_orientations
    )

    ccd(chain_positions, chain_orientations, chain_offsets, target_pose)

    apply_ik(
        meta_data,
        joint_positions,
        joint_orientations,
        chain_positions,
        chain_orientations,
    )

    return joint_positions, joint_orientations


def bonus_inverse_kinematics(
    meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose
):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """

    chain_positions, chain_orientations, chain_offsets = get_chain_list(
        meta_data, joint_positions, joint_orientations
    )
    ccd(chain_positions, chain_orientations, chain_offsets, left_target_pose)

    apply_ik(
        meta_data,
        joint_positions,
        joint_orientations,
        chain_positions,
        chain_orientations,
    )
    
    meta_data.root_joint = 'rToeJoint_end'
    meta_data.end_joint = 'rWrist_end'

    chain_positions, chain_orientations, chain_offsets = get_chain_list(
        meta_data, joint_positions, joint_orientations
    )
    ccd(chain_positions, chain_orientations, chain_offsets, right_target_pose)

    apply_ik(
        meta_data,
        joint_positions,
        joint_orientations,
        chain_positions,
        chain_orientations,
    )
    return joint_positions, joint_orientations
