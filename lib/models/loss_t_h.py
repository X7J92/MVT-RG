
import torch
import torch
import torch.nn.functional as F
import torch.nn as nn

#
# def compute_cmpm_loss(video_embeddings, text_embeddings, labels, epsilon=1e-8):
#     """
#     Cross-Modal Projection Matching Loss(CMPM) - 双向损失
#     :param video_embeddings: Tensor of shape (batch_size, embedding_dim)
#     :param text_embeddings: Tensor of shape (num_texts, embedding_dim)
#     :param labels: Tensor of shape (batch_size,) with 1 and 0
#     :param epsilon: A small constant for numerical stability
#     :return:
#         cmpm_loss: overall cmpm loss
#         pos_avg_sim: average cosine-similarity for positive pairs
#         neg_avg_sim: average cosine-similarity for negative pairs
#     """
#     batch_size = video_embeddings.shape[0]
#     num_texts = text_embeddings.shape[0]  # 文本数量
#
#     # 归一化文本嵌入和视频嵌入
#     text_embeddings_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
#     # video_embeddings_norm = video_embeddings / video_embeddings.norm(dim=1, keepdim=True)
#
#     # 计算视频嵌入与文本嵌入的余弦相似度
#
#     sim_video_to_text = torch.matmul(video_embeddings, text_embeddings_norm.T)  # 形状为 (16, 4)
#
#     b=sim_video_to_text.shape[1]
#
#
#     # 对第二个维度进行累加，保留第一个维度
#     sum_tensor = torch.sum(sim_video_to_text, dim=1, keepdim=True)
#
#     sim_video_to_text = sum_tensor / b
#     # print(sim_video_to_text.shape)
#     sim_text_to_video = sim_video_to_text.T  # 形状为 (4, 16)，即文本到视频的相似度
#
#     # 扩展标签维度，labels扩展为 (16, 4) 形状，与 sim_video_to_text 匹配
#     labels_mask = labels.unsqueeze(1).expand(batch_size, 1).float()
#
#     # 归一化标签
#     labels_mask_norm = labels_mask / labels_mask.sum(dim=0, keepdim=True)
#     # print(labels_mask_norm)
#     labels_mask_norm = labels_mask_norm.cuda()
#     # 计算视频到文本的投影分布
#     video_to_text_pred = F.softmax(sim_video_to_text, dim=1)  # 对文本的维度进行 softmax
#     video_to_text_loss = video_to_text_pred * (
#             F.log_softmax(sim_video_to_text, dim=1) - torch.log(labels_mask_norm + epsilon)
#     )
#
#     # 计算文本到视频的投影分布
#     text_to_video_pred = F.softmax(sim_text_to_video, dim=1)  # 对视频的维度进行 softmax
#     text_to_video_loss = text_to_video_pred * (
#             F.log_softmax(sim_text_to_video, dim=1) - torch.log(labels_mask_norm.T + epsilon)
#     )
#
#     # CMPM 损失为投影损失的和（双向）
#     cmpm_loss = -(torch.mean(torch.sum(video_to_text_loss, dim=1)) + torch.mean(torch.sum(text_to_video_loss, dim=1)))*0.5
#
#
#
#     return cmpm_loss
#

#
# def triplet_loss(video_features, text_features, mask, margin):
#     # 扩展文本特征维度以匹配视频特征，以便在512特征维度上计算余弦相似度
#     text_features_ext = text_features.unsqueeze(1)  # shape (4, 1, 512)
#     video_features_ext = video_features.unsqueeze(0)  # shape (1, 16, 512)
#
#     # 计算余弦相似度，维度2（512）是特征维度
#     sims = F.cosine_similarity(text_features_ext, video_features_ext, dim=2)  # 结果 shape (4, 16)
#
#     # 使用mask找到每个文本对应的正样本的索引
#     positive_indices = mask.nonzero(as_tuple=True)[0]
#     positive_scores = sims[torch.arange(sims.size(0)), positive_indices]
#
#     # 计算所有负样本的分数
#     negative_scores = sims.masked_fill(mask.bool().unsqueeze(0), -float('inf'))
#     max_negative_scores, _ = negative_scores.max(dim=1)  # 最大的负样本分数
#
#     # 计算三元组损失
#     losses = F.relu(margin + max_negative_scores - positive_scores)  # 使用ReLU保证损失非负
#     return losses.mean()
#
#
# def compute_cmpm_loss(video_features, text_features, mask, logit_scale):
#         mask=mask.cuda()
#
#         # video_features = video_features / video_features.norm(dim=-1, keepdim=True)
#         # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
#         margin = 0.2
#         loss_tri=triplet_loss(video_features, text_features, mask, margin)
#
#         # 扩展视频特征以匹配文本特征
#         video_features_ext = video_features.unsqueeze(0)  # shape becomes (1, 16, 512)
#         text_features_ext = text_features.unsqueeze(1)  # shape becomes (4, 1, 512)
#
#         # 计算余弦相似度，应在最后一个维度上执行
#         sims = F.cosine_similarity(text_features_ext, video_features_ext, dim=2)  # 结果 shape (4, 16)
#         # print(sims.shape)
#         # 应用掩码和logit_scale
#         masked_sims = sims * mask * logit_scale.exp()
#
#         # 计算log softmax，注意维度
#         t2v_log_sm = F.log_softmax(masked_sims, dim=1)
#         v2t_log_sm = F.log_softmax(masked_sims.transpose(0, 1), dim=1)
#
#         # 提取对应匹配的损失，平均后取负
#         t2v_neg_ce = t2v_log_sm.diag().mean()
#         v2t_neg_ce = v2t_log_sm.diag().mean()
#         loss_n=-(t2v_neg_ce + v2t_neg_ce) / 2
#         # 返回损失的平均值
#         return loss_n + loss_tri




loss_fn = nn.CrossEntropyLoss()
def compute_cmpm_loss(video_features, text_features, mask, logit_scale):
    """
    计算视频特征和文本特征之间的损失。

    参数:
        video_features (torch.Tensor): 视频特征张量，形状为 (batch_size, feature_dim)
        text_features (torch.Tensor): 文本特征张量，形状为 (num_texts, feature_dim)
        mask (torch.Tensor): 掩码张量，形状为 (batch_size, 1)，表示匹配情况

    返回:
        float: 计算得到的损失值
    """
    # 归一化文本特征
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    video_features=video_features/video_features.norm(dim=-1, keepdim=True)
    # 转置文本特征以用于矩阵乘法
    text_features_T = text_features.permute(1, 0)

    # 计算视频和文本之间的相似度
    sims = torch.matmul(video_features, text_features_T)

    # 计算相似度的平均值
    number = text_features.shape[0]
    sum_tensor = torch.sum(sims, dim=1, keepdim=True)
    sims = sum_tensor / number
    mask=mask.unsqueeze(1).cuda()
    # 使用BCEWithLogitsLoss计算损失
    # 将相似度转换为概率分布
    sims = F.softmax(sims, dim=0)
    # print(sims)
    loss =  F.binary_cross_entropy_with_logits(sims, mask)

    return loss*2











def loss_t_h(batch_output_tensor,vg_hs_video,sentence_number,logit_scale):

    # print(batch_output_tensor.shape)torch.Size([16, 8, 8, 512])
    # print(vg_hs_video.shape)torch.Size([16, 512])
    # print(sentence_number.shape)torch.Size([16, 1])

    sentence_number = sentence_number.squeeze(1)
    batch_size = batch_output_tensor.shape[0]
    loss_IVC_list=[]
    loss_NCE_list=[]
    for i in range(batch_size):
     if sentence_number[i]==2:
        loss_layer_nce2=[]
        batch_output_tensor2=batch_output_tensor[i,:,:,:]#取出批次中的段落文本
        for j in range(2):
            batch_output_tensor_layer=batch_output_tensor2[:,j,:]
            batch_output_tensor_layer_mask=batch_output_tensor_layer[:2, :]
            mask = torch.zeros(batch_size)
            mask[i] = 1
            cmpm_loss = compute_cmpm_loss(vg_hs_video,batch_output_tensor_layer_mask, mask,logit_scale)
            # print(cmpm_loss)
            loss_layer_nce2.append(cmpm_loss)

        loss_IVC_batch2 = (torch.max(loss_layer_nce2[0] - loss_layer_nce2[1] + 0.50, 0)[0])
        loss_NCE_batch2 = (loss_layer_nce2[1] + loss_layer_nce2[0])*0.5
        loss_IVC_list.append(loss_IVC_batch2)
        loss_NCE_list.append(loss_NCE_batch2)

     elif sentence_number[i]==3:
         loss_layer_nce3 = []
         batch_output_tensor3 = batch_output_tensor[i, :, :, :]  # 取出批次中的段落文本
         for j in range(3):
             batch_output_tensor_layer = batch_output_tensor3[:, j, :]
             batch_output_tensor_layer_mask = batch_output_tensor_layer[:3, :]
             mask = torch.zeros(batch_size)
             mask[i] = 1
             cmpm_loss = compute_cmpm_loss(vg_hs_video, batch_output_tensor_layer_mask, mask,logit_scale)
             loss_layer_nce3.append(cmpm_loss)

         loss_IVC_batch3 = (torch.max(loss_layer_nce3[2] - loss_layer_nce3[1] + 0.50, 0)[0] + torch.max(loss_layer_nce3[1] - loss_layer_nce3[0] + 0.50, 0)[0])*1/2
         loss_NCE_batch3 = (loss_layer_nce3[0] + loss_layer_nce3[1]+loss_layer_nce3[2])*1/3
         loss_IVC_list.append(loss_IVC_batch3)
         loss_NCE_list.append(loss_NCE_batch3)

     elif sentence_number[i]==4:
         loss_layer_nce4 = []
         batch_output_tensor4 = batch_output_tensor[i, :, :, :]  # 取出批次中的段落文本
         for j in range(4):
            batch_output_tensor_layer = batch_output_tensor4[:, j, :]
            batch_output_tensor_layer_mask = batch_output_tensor_layer[:4, :]
            mask = torch.zeros(batch_size)
            mask[i] = 1
            cmpm_loss = compute_cmpm_loss(vg_hs_video, batch_output_tensor_layer_mask, mask,logit_scale)
            loss_layer_nce4.append(cmpm_loss)

         loss_IVC_batch4 = (torch.max(loss_layer_nce4[3] - loss_layer_nce4[2] + 0.50, 0)[0] + torch.max(loss_layer_nce4[2] - loss_layer_nce4[1] + 0.50, 0)[0]+torch.max(loss_layer_nce4[1] - loss_layer_nce4[0] + 0.50, 0)[0]) * 1 / 3
         loss_NCE_batch4 = (loss_layer_nce4[0] + loss_layer_nce4[1] + loss_layer_nce4[2]+loss_layer_nce4[3]) * 1 / 4
         loss_IVC_list.append(loss_IVC_batch4)
         loss_NCE_list.append(loss_NCE_batch4)


     elif sentence_number[i]==5:
        loss_layer_nce5 = []
        batch_output_tensor5 = batch_output_tensor[i, :, :, :]  # 取出批次中的段落文本
        for j in range(5):
            batch_output_tensor_layer = batch_output_tensor5[:, j, :]
            batch_output_tensor_layer_mask = batch_output_tensor_layer[:5, :]
            mask = torch.zeros(batch_size)
            mask[i] = 1
            cmpm_loss = compute_cmpm_loss(vg_hs_video, batch_output_tensor_layer_mask, mask,logit_scale)
            loss_layer_nce5.append(cmpm_loss)

        loss_IVC_batch5 = (torch.max(loss_layer_nce5[4] - loss_layer_nce5[3] + 0.50, 0)[0] +
                           torch.max(loss_layer_nce5[3] - loss_layer_nce5[2] + 0.50, 0)[0] +
                           torch.max(loss_layer_nce5[2] - loss_layer_nce5[1] + 0.50, 0)[0] +
                           torch.max(loss_layer_nce5[1] - loss_layer_nce5[0] + 0.50, 0)[0]) * 1 / 4
        loss_NCE_batch5 = (loss_layer_nce5[0] + loss_layer_nce5[1] + loss_layer_nce5[2] + loss_layer_nce5[3]+loss_layer_nce5[4]) * 1 / 5
        loss_IVC_list.append(loss_IVC_batch5)
        loss_NCE_list.append(loss_NCE_batch5)

     elif sentence_number[i]==6:
        loss_layer_nce6 = []
        batch_output_tensor6 = batch_output_tensor[i, :, :, :]  # 取出批次中的段落文本
        for j in range(6):
            batch_output_tensor_layer = batch_output_tensor6[:, j, :]
            batch_output_tensor_layer_mask = batch_output_tensor_layer[:6, :]
            mask = torch.zeros(batch_size)
            mask[i] = 1
            cmpm_loss = compute_cmpm_loss(vg_hs_video, batch_output_tensor_layer_mask, mask,logit_scale)
            loss_layer_nce6.append(cmpm_loss)

        loss_IVC_batch6 = (torch.max(loss_layer_nce6[5] - loss_layer_nce6[4] + 0.50, 0)[0] +
                           torch.max(loss_layer_nce6[4] - loss_layer_nce6[3] + 0.50, 0)[0] +
                           torch.max(loss_layer_nce6[3] - loss_layer_nce6[2] + 0.50, 0)[0] +
                           torch.max(loss_layer_nce6[2] - loss_layer_nce6[1] + 0.50, 0)[0] +
                           torch.max(loss_layer_nce6[1] - loss_layer_nce6[0] + 0.50, 0)[0]) * 1 / 5
        loss_NCE_batch6 = (loss_layer_nce6[0] + loss_layer_nce6[1] + loss_layer_nce6[2] + loss_layer_nce6[3]+loss_layer_nce6[4]+loss_layer_nce6[5]) * 1 / 6
        loss_IVC_list.append(loss_IVC_batch6)
        loss_NCE_list.append(loss_NCE_batch6)

     elif sentence_number[i]==7:
        loss_layer_nce7 = []
        batch_output_tensor7 = batch_output_tensor[i, :, :, :]  # 取出批次中的段落文本
        for j in range(7):
            batch_output_tensor_layer = batch_output_tensor7[:, j, :]
            batch_output_tensor_layer_mask = batch_output_tensor_layer[:7, :]
            mask = torch.zeros(batch_size)
            mask[i] = 1
            cmpm_loss = compute_cmpm_loss(vg_hs_video, batch_output_tensor_layer_mask, mask,logit_scale)
            loss_layer_nce7.append(cmpm_loss)

        loss_IVC_batch7 = (torch.max(loss_layer_nce7[6] - loss_layer_nce7[5] + 0.50, 0)[0] +
                           torch.max(loss_layer_nce7[5] - loss_layer_nce7[4] + 0.50, 0)[0] +
                           torch.max(loss_layer_nce7[4] - loss_layer_nce7[3] + 0.50, 0)[0] +
                           torch.max(loss_layer_nce7[3] - loss_layer_nce7[2] + 0.50, 0)[0] +
                           torch.max(loss_layer_nce7[2] - loss_layer_nce7[1] + 0.50, 0)[0] +
                           torch.max(loss_layer_nce7[1] - loss_layer_nce7[0] + 0.50, 0)[0]) * 1 / 6
        loss_NCE_batch7 = (loss_layer_nce7[0] + loss_layer_nce7[1] + loss_layer_nce7[2] + loss_layer_nce7[3]+loss_layer_nce7[4]+loss_layer_nce7[5]+loss_layer_nce7[6]) * 1 / 7
        loss_IVC_list.append(loss_IVC_batch7)
        loss_NCE_list.append(loss_NCE_batch7)

     elif sentence_number[i]==8:
        loss_layer_nce8 = []
        batch_output_tensor8 = batch_output_tensor[i, :, :, :]  # 取出批次中的段落文本
        for j in range(8):
            batch_output_tensor_layer = batch_output_tensor8[:, j, :]
            batch_output_tensor_layer_mask = batch_output_tensor_layer[:8, :]
            mask = torch.zeros(batch_size)
            mask[i] = 1
            cmpm_loss = compute_cmpm_loss(vg_hs_video, batch_output_tensor_layer_mask, mask,logit_scale)
            loss_layer_nce8.append(cmpm_loss)

        loss_IVC_batch8 = (torch.max(loss_layer_nce8[7] - loss_layer_nce8[6] + 0.50, 0)[0] +
                           torch.max(loss_layer_nce8[6] - loss_layer_nce8[5] + 0.50, 0)[0] +
                           torch.max(loss_layer_nce8[5] - loss_layer_nce8[4] + 0.50, 0)[0] +
                           torch.max(loss_layer_nce8[4] - loss_layer_nce8[3] + 0.50, 0)[0] +
                           torch.max(loss_layer_nce8[3] - loss_layer_nce8[2] + 0.50, 0)[0] +
                           torch.max(loss_layer_nce8[2] - loss_layer_nce8[1] + 0.50, 0)[0] +
                           torch.max(loss_layer_nce8[1] - loss_layer_nce8[0] + 0.50, 0)[0]) * 1 / 7
        loss_NCE_batch8 = (loss_layer_nce8[0] + loss_layer_nce8[1] + loss_layer_nce8[2] + loss_layer_nce8[3]+loss_layer_nce8[4]+loss_layer_nce8[5]+loss_layer_nce8[6]) * 1 / 8
        loss_NCE_list.append(loss_NCE_batch8)
        loss_IVC_list.append(loss_IVC_batch8)

    loss_NCE=sum(loss_NCE_list)/len(loss_NCE_list)
    loss_IVC = sum(loss_IVC_list) / len(loss_IVC_list)
    return loss_NCE,loss_IVC
