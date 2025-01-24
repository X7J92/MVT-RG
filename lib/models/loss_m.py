import torch
import random

import torch
import random


def sample_and_calculate_orthogonality_loss(tensors):
    """
    从包含多个矩阵的 tensor 中为每个矩阵随机选择另一个不重复的矩阵，
    计算正交损失，并规范化。

    参数:
    - tensors (torch.Tensor): 形状为 (N, 32, 32) 的 tensor，包含 N 个矩阵

    返回:
    - normalized_loss (torch.Tensor): 规范化后的正交损失
    """
    N = tensors.size(0)  # 获取 tensor 的第一个维度大小，即矩阵的数量
    if N < 2:
        raise ValueError("至少需要两个矩阵来计算正交损失")

    # 创建一个索引列表并随机洗牌
    indices = list(range(N))
    random.shuffle(indices)

    # 形成矩阵对并计算损失
    total_loss = 0.0
    for i in range(N):
        # 选择第 i 个矩阵和随机选择的另一个矩阵形成一对
        j = (i + 1) % N  # 确保每个矩阵都参与一次，并且不与自身配对
        A = tensors[indices[i]]
        B = tensors[indices[j]]
        C = A*B
        # 计算正交损失A.T @ B
        loss = (C).norm('fro') ** 2
        total_loss += loss

    # 规范化损失
    normalized_loss = total_loss / N
    return normalized_loss


#
# import torch
# import random
#
# def orthogonality_loss(A, B):
#     """计算两个矩阵之间的正交损失"""
#     if A.dim() == 3:
#         A = A.permute(0, 2, 1)
#     else:
#         A = A.T
#     return (A @ B).norm('fro') ** 2
#
# def sample_and_calculate_orthogonality_loss(tensors):
#     """从包含多个矩阵的 tensor 中随机选择两个矩阵，并计算它们之间的正交损失。"""
#     N = tensors.size(0)
#     if N < 2:
#         raise ValueError("至少需要两个矩阵来计算正交损失")
#     idxs = random.sample(range(N), 2)
#     A = tensors[idxs[0]]
#     B = tensors[idxs[1]]
#     return orthogonality_loss(A, B) / N

def apply_mask_and_calculate_loss(prediction, map_mask, sentence_mask):
    """应用掩码到预测上，并计算每个批次中有效句子的正交性损失。"""
    masked_prediction = prediction * map_mask
    total_loss = 0.0

    for i in range(prediction.shape[0]):  # 直接使用 prediction.shape[0] 获取批次数量
        valid_indices = sentence_mask[i].squeeze(-1).nonzero(as_tuple=True)[0]
        if len(valid_indices) > 1:
            valid_tensors = masked_prediction[i][valid_indices]
            valid_tensors = valid_tensors.reshape(-1, 32, 32)
            loss = sample_and_calculate_orthogonality_loss(valid_tensors)
            total_loss += loss

    average_loss = total_loss / prediction.shape[0]
    return average_loss

# # 示例使用
# prediction = torch.randn(16, 8, 1, 32, 32)
# map_mask = torch.ones(16, 8, 1, 32, 32)
# sentence_mask = torch.randint(0, 2, (16, 8, 1))
#
# average_loss = apply_mask_and_calculate_loss(prediction, map_mask, sentence_mask)
# print("Average Loss per Batch:", average_loss)
