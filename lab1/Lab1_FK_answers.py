import numpy as np
from scipy.spatial.transform import Rotation as R


def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith("Frame Time"):
                break
        motion_data = []
        for line in lines[i + 1 :]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1, -1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def load_joint_data(lines, i, parent, level, joint_name, joint_parent, joint_offset):
    info = lines[i].split()
    if joint_name.__contains__(info[1]):
        return
    joint_name.append(info[1])
    joint_parent.append(parent)
    datas = lines[i + 2].split()
    joint_offset.append([datas[1], datas[2], datas[3]])
    parentIndex = len(joint_name) - 1
    i += 4
    nextInfo = lines[i].split()
    if nextInfo[0] == "End":
        joint_name.append(joint_name[parentIndex] + "_end")
        joint_parent.append(parentIndex)
        datas = lines[i + 2].split()
        joint_offset.append([datas[1], datas[2], datas[3]])
        # return
    elif nextInfo[0] == "JOINT":
        load_joint_data(
            lines, i, parentIndex, level + 1, joint_name, joint_parent, joint_offset
        )

    squard = "    " * level
    for j in range(i + 4, len(lines)):
        if lines[j].startswith(squard + "JOINT"):
            if parent == -1:
                parent = 0
            load_joint_data(
                lines, j, parent, level, joint_name, joint_parent, joint_offset
            )
        elif lines[j].startswith("    " * (level - 1) + "}"):
            return


def load_rest_pose(bvh_file_path):
    joint_name, joint_parent, joint_offset = part1_calculate_T_pose(bvh_file_path)
    length = len(joint_name)
    joint_positions = np.zeros((length, 3), dtype=np.float64)
    joint_orientations = np.zeros((length, 4), dtype=np.float64)
    for i in range(length):
        if joint_parent[i] == -1:
            joint_positions[i] = joint_offset[i]
        else:
            joint_positions[i] = joint_positions[joint_parent[i]] + joint_offset[i]
        if joint_name[i] == "lShoulder":
            joint_orientations[i] = R.from_euler(
                "XYZ", [0, 0, -45], degrees=True
            ).as_quat()
        elif joint_name[i] == "rShoulder":
            joint_orientations[i] = R.from_euler(
                "XYZ", [0, 0, 45], degrees=True
            ).as_quat()
        else:
            joint_orientations[i, 3] = 1.0

    return joint_positions, joint_orientations


def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = []
    with open(bvh_file_path, "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith("ROOT"):
                load_joint_data(lines, i, -1, 1, joint_name, joint_parent, joint_offset)
                break
    joint_offset = np.array(joint_offset, dtype=np.float64)
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(
    joint_name, joint_parent, joint_offset, motion_data, frame_id
):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    frame_data = motion_data[frame_id]
    joint_positions = []
    joint_orientations = []
    rootPos = [frame_data[0], frame_data[1], frame_data[2]]
    rootRotation = R.from_euler(
        "XYZ", [frame_data[3], frame_data[4], frame_data[5]], degrees=True
    )

    joint_positions.append(joint_offset[0] + rootPos)
    joint_orientations.append(rootRotation.as_quat())
    jointIndex = 1
    for i in range(1, len(joint_name)):
        parent = joint_parent[i]
        parent_quat = R.from_quat(joint_orientations[parent])
        joint_positions.append(
            joint_positions[parent] + parent_quat.apply(joint_offset[i])
        )

        if not joint_name[i].endswith("_end"):
            local_euler = [
                frame_data[(jointIndex + 1) * 3],
                frame_data[(jointIndex + 1) * 3 + 1],
                frame_data[(jointIndex + 1) * 3 + 2],
            ]
            r_1 = R.from_euler("XYZ", local_euler, degrees=True)
            joint_orientations.append((parent_quat * r_1).as_quat())
            jointIndex += 1
        else:
            joint_orientations.append([0, 0, 0, 1])

    joint_positions = np.array(joint_positions, dtype=np.float64)
    joint_orientations = np.array(joint_orientations, dtype=np.float64)
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出:
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    temp_motion_data = load_motion_data(A_pose_bvh_path)
    
    A_joint_name, _, _ = part1_calculate_T_pose(A_pose_bvh_path)
    T_joint_name, _, _ = part1_calculate_T_pose(T_pose_bvh_path)
    
    # 储存 A_pose 骨骼的索引映射
    A_name2index = {}
    index = 0
    for A_name in A_joint_name:
        if ("_end" not in A_name):
            index += 1
            A_name2index[A_name] = index
    
    # 使用 A_pose 骨骼的索引定位 A_pose 骨骼动画数据，从而映射到相应 T_pose 中的骨骼上
    motion_data = []
    for temp_frame_data in temp_motion_data:
        frame_data = []
        frame_data.append(temp_frame_data[0:3])
        for T_name in T_joint_name:
            if ("_end" not in T_name):
                frame_data.append(temp_frame_data[A_name2index[T_name] * 3 : A_name2index[T_name] * 3 + 3])
                # 对肩膀部做特殊的选择，-1表示最后一个元素
                if (T_name == "lShoulder"):
                    frame_data[-1][-1] -= 45
                elif (T_name == "rShoulder"):
                    frame_data[-1][-1] += 45
        motion_data.append(np.array(frame_data).reshape(1,-1))

    motion_data = np.concatenate(motion_data, axis=0)
    return motion_data
