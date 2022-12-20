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
        return
    elif nextInfo[0] == "JOINT":
        load_joint_data(
            lines, i, parentIndex, level + 1, joint_name, joint_parent, joint_offset
        )

    squard = "    " * level
    for j in range(i + 1, len(lines)):
        if lines[j].startswith(squard + "JOINT"):
            if parent == -1:
                parent = 0
            load_joint_data(
                lines, j, parent, level, joint_name, joint_parent, joint_offset
            )
        elif lines[j].startswith("    " * (level - 1) + "}"):
            break


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
    joint_positions = None
    joint_orientations = None
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
    motion_data = None
    return motion_data
