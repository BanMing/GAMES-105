from sys import flags
from bvh_utils import *


# ---------------你的代码------------------#
# translation 和 orientation 都是全局的
def skinning(
    joint_translation,
    joint_orientation,
    T_pose_joint_translation,
    T_pose_vertex_translation,
    skinning_idx,
    skinning_weight,
):
    """
    skinning函数，给出一桢骨骼的位姿，计算蒙皮顶点的位置
    假设M个关节，N个蒙皮顶点，每个顶点受到最多4个关节影响
    输入：
        joint_translation: (M,3)的ndarray, 目标关节的位置
        joint_orientation: (M,4)的ndarray, 目标关节的旋转，用四元数表示
        T_pose_joint_translation: (M,3)的ndarray, T pose下关节的位置
        T_pose_vertex_translation: (N,3)的ndarray, T pose下蒙皮顶点的位置
        skinning_idx: (N,4)的ndarray, 每个顶点受到哪些关节的影响（假设最多受4个关节影响）
        skinning_weight: (N,4)的ndarray, 每个顶点受到对应关节影响的权重
    输出：
        vertex_translation: (N,3)的ndarray, 蒙皮顶点的位置
    """
    vertex_translation = T_pose_vertex_translation.copy()

    # ---------------你的代码------------------#
    vertex_len = len(T_pose_vertex_translation)
    for i in range(vertex_len):
        vertex_translation[i][:] = 0
        for j in range(4):
            cur_vertex_joint = skinning_idx[i][j]
            cur_vertex_weight = skinning_weight[i][j]
            cur_vertex_joint = skinning_idx[i][j]
            Q_j = R.from_quat(joint_orientation[cur_vertex_joint])
            o_j = joint_translation[cur_vertex_joint]
            r_ij = (
                T_pose_vertex_translation[i]
                - T_pose_joint_translation[cur_vertex_joint]
            )
            
            vertex_translation[i] += cur_vertex_weight * (Q_j.apply(r_ij) + o_j)

    return vertex_translation
