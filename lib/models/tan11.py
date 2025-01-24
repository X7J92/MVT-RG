# # from torch import nn
# # from lib.core.config import config
# # import lib.models.frame_modules as frame_modules
# # import lib.models.prop_modules as prop_modules
# # import lib.models.map_modules as map_modules
# # import lib.models.fusion_modules as fusion_modules
# # import lib.models.bmn_modules as bmn_layer
# # from IPython import embed
# #
# # import math
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import matplotlib.pyplot as plt
# # from .vl_transformer import build_vl_transformer
# #
# # from lib.models.transformer import DualTransformer
# #
# #
# #
# # import torch
# # import torch.nn.functional as F
# # from torch_geometric.nn import GCNConv
# # import random
# # import time
# #
# # class DynamicGCNFusion(torch.nn.Module):
# #     def __init__(self):
# #         super(DynamicGCNFusion, self).__init__()
# #         self.input_dim = 512  # 直接定义input_dim
# #         self.hidden_dim = 512  # 直接定义hidden_dim
# #         self.conv1 = GCNConv(self.input_dim, self.hidden_dim)
# #         self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
# #
# #     def forward(self, x, edge_index):
# #         # 图卷积层1
# #         x = self.conv1(x, edge_index)
# #         x = F.relu(x)
# #         # 图卷积层2
# #         x = self.conv2(x, edge_index)
# #         return x
# #
# #
# # def generate_group_layers(ids):
# #     start_time = time.time()
# #     layers = {1: [[id] for id in ids]}
# #     total_levels = len(ids)
# #
# #     for level in range(2, total_levels):
# #         while True:
# #             current_layer = []
# #             all_possible_combinations = set()
# #
# #             for base_group in layers[level - 1]:
# #                 available_ids = [id for id in ids if id not in base_group]
# #                 random.shuffle(available_ids)
# #                 for chosen_id in available_ids:
# #                     new_group = base_group + [chosen_id]
# #                     new_group_sorted_tuple = tuple(sorted(new_group))
# #                     if new_group_sorted_tuple not in all_possible_combinations:
# #                         all_possible_combinations.add(new_group_sorted_tuple)
# #                         current_layer.append(new_group)
# #                         break
# #
# #             if len(current_layer) == len(ids):
# #                 layers[level] = current_layer
# #                 break
# #
# #             if time.time() - start_time > 0.0001:  # 改进了超时时间为1秒
# #                 return None
# #
# #     last_layer = []
# #     layer_last_groups = layers[total_levels - 1]
# #     for base_group in layer_last_groups:
# #         available_ids = [id for id in ids if id not in base_group]
# #         random.shuffle(available_ids)
# #         for chosen_id in available_ids:
# #             new_group = base_group + [chosen_id]
# #             new_group_sorted_tuple = tuple(sorted(new_group))
# #             if new_group_sorted_tuple not in last_layer:
# #                 last_layer.append(new_group)
# #                 if len(last_layer) == len(ids):
# #                     break
# #         if len(last_layer) == len(ids):
# #             break
# #
# #     layers[total_levels] = last_layer
# #     return layers
# #
# # def convert_layers_to_list(layers):
# #     return [[list(map(int, group)) for group in layers[level]] for level in sorted(layers)]
# #
# # def create_edge_index(groups):
# #     src, dst = [], []
# #     for group in groups:
# #         for i in range(len(group)):
# #             for j in range(len(group)):
# #                 if i != j:
# #                     src.append(group[i])
# #                     dst.append(group[j])
# #     return torch.tensor([src, dst], dtype=torch.long)
# #
# # def iou(pred, gt):
# #     assert isinstance(pred, list) and isinstance(gt, list)
# #     pred_is_list = isinstance(pred[0], list)
# #     gt_is_list = isinstance(gt[0], list)
# #     if not pred_is_list: pred = [pred]
# #     if not gt_is_list: gt = [gt]
# #     pred, gt = np.array(pred), np.array(gt)
# #     inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
# #     inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
# #     inter = np.maximum(0.0, inter_right - inter_left)
# #     union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
# #     union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
# #     union = np.maximum(0.0, union_right - union_left)
# #     # 避免除以零：在分母上加一个很小的数
# #     overlap = 1.0 * inter / (union + 1e-6)
# #     if not gt_is_list:
# #         overlap = overlap[:, 0]
# #     if not pred_is_list:
# #         overlap = overlap[0]
# #     return overlap
# #
# #
# # def calculate_overlaps(duration_tensor, y_t, num_clips, num_sentences, batch_size):
# #     """
# #     Calculate overlaps between time segments and given time points.
# #
# #     Args:
# #         duration_tensor (Tensor): Tensor containing durations for each batch.
# #         y_t (Tensor): Tensor containing start and end times for sentences.
# #         num_clips (int): Number of time segments.
# #         num_sentences (int): Number of sentences.
# #         batch_size (int): Size of each batch.
# #
# #     Returns:
# #         Tensor: A tensor containing overlap values.
# #     """
# #
# #     overlaps_list = []
# #
# #     for batch in range(batch_size):
# #         overlaps_batch = []
# #         duration = duration_tensor[batch].item()  # 获取当前批次的持续时间
# #
# #         for sentence in range(num_sentences):
# #             gt_s_time, gt_e_time = y_t[batch, sentence, 0, 0].item(), y_t[batch, sentence, 0, 1].item()
# #
# #             if gt_s_time == 0 and gt_e_time == 0:
# #                 overlaps_batch.append(torch.zeros(1, num_clips, num_clips))
# #                 continue
# #
# #             s_times = torch.arange(0, num_clips).float() * duration / num_clips
# #             e_times = torch.arange(1, num_clips + 1).float() * duration / num_clips
# #             overlaps = iou(torch.stack([s_times[:, None].expand(-1, num_clips),
# #                                         e_times[None, :].expand(num_clips, -1)], dim=2).view(-1, 2).tolist(),
# #                            [gt_s_time, gt_e_time]).reshape(num_clips, num_clips)
# #
# #             overlaps_batch.append(torch.from_numpy(overlaps).unsqueeze(0))
# #
# #         overlaps_list.append(torch.cat(overlaps_batch, 0))
# #
# #     return torch.stack(overlaps_list, 0)
# #
# #
# # def get_max_score_one_timestamps(scores, durations):
# #     # 假设scores的维度为[4, 8, 1, 32, 32]
# #     # 初始化输出时间戳列表，确保维度为 (4, 8, 3, 2) 来存储前三个最大值的时间戳
# #     out_max_timestamps = torch.zeros(scores.size(0), scores.size(1), 1, 2)
# #     # 初始化输出分数列表，确保维度为 (4, 8, 3) 来存储前三个最大值的分数
# #     out_max_scores = torch.zeros(scores.size(0), scores.size(1), 1)
# #
# #     for batch_idx, (score_sent, duration) in enumerate(zip(scores, durations)):
# #         for text_idx, score in enumerate(score_sent.squeeze()):  # 去掉维度为1的维度
# #             # 如果当前分数矩阵不全为0
# #             if score.sum() > 1e-3:
# #                 T = score.shape[-1]
# #                 # 将score展平并获取前三个最大值及其索引
# #                 flat_score = score.view(-1)
# #                 top1_values, top1_max_indices = torch.topk(flat_score, 1)  # 获取前三个最大值及其索引
# #
# #                 for i, max_index in enumerate(top1_max_indices):
# #                     max_position = np.unravel_index(max_index.item(), (T, T))
# #                     # 计算时间
# #                     target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
# #                     max_timestamp = (np.array(max_position).astype(float) / target_size * duration)
# #                     # 更新输出时间戳列表
# #                     out_max_timestamps[batch_idx, text_idx, i] = torch.tensor(max_timestamp)
# #                     # 更新输出分数列表
# #                     out_max_scores[batch_idx, text_idx, i] = top1_values[i]
# #
# #     # 返回包含时间戳和分数的元组
# #     return out_max_timestamps, out_max_scores
# #
# #
# # def get_positional_encoding(d_model, idx):
# #     positional_encoding = torch.zeros((d_model,))  # (max_length, d_model)
# #     i = idx
# #     for j in range(d_model):
# #         if j % 2 == 0:
# #             positional_encoding[j] = math.sin(i / math.pow(10000, j / d_model))
# #         else:
# #             positional_encoding[j] = math.cos(i / math.pow(10000, (j - 1) / d_model))
# #
# # #     positional_encoding = positional_encoding.unsqueeze(0)  # (1, max_length, d_model)
# #
# #     return positional_encoding
# #
# # def get_proposal_results(scores, durations):
# #         # assume all valid scores are larger than one
# #         out_sorted_times = []
# #         for score_sent, duration in zip(scores, durations):
# #             sent_times = []
# #             for score in score_sent:
# #                 if score.sum() < 1e-3:
# #                     break
# #                 T = score.shape[-1]
# #                 sorted_indexs = np.dstack(
# #                     np.unravel_index(np.argsort(score.cpu().detach().numpy().ravel())[::-1], (T, T))).tolist()
# #                 sorted_indexs = np.array([item for item in sorted_indexs[0] if item[0] <= item[1]]).astype(float)
# #
# #                 sorted_indexs[:, 1] = sorted_indexs[:, 1] + 1
# #                 sorted_indexs = torch.from_numpy(sorted_indexs).cuda()
# #                 target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
# #                 sent_times.append((sorted_indexs.float() / target_size * duration).tolist())
# #             out_sorted_times.append(sent_times)
# #         return out_sorted_times
# #
# # #
# # def _generate_proposals_feat(frames_feat, props, duration):
# #     props_feats = []
# #     # props_len = []
# #
# #     for f, p, d in zip(frames_feat, props, duration):
# #         total_frames = f.shape[0]
# #
# #         for s, e in p:
# #             if s == 0 and e == 0 and d==0:
# #                 # 如果 s 和 e 都是 0，则随机初始化特征
# #                 # random_feat = torch.normal(0,0.1,size=(16,512)).cuda()
# #                 random_feat = torch.zeros(16, 500).cuda()
# #                 props_feats.append(random_feat)
# #                 # props_len.append(batch_size * 8)
# #                 continue
# #
# #             # 将 s 和 e 映射到相对于视频持续时间的比例，并转换为帧索引
# #             s_frame = int((s / d) * total_frames)
# #             e_frame = int((e / d) * total_frames)
# #             s_frame, e_frame = min(s_frame, total_frames - 1), min(e_frame, total_frames - 1)
# #
# #             # 生成索引并提取特征
# #             idx = np.linspace(start=s_frame, stop=e_frame, num=16, endpoint=False).astype(np.int32)
# #             idx = np.minimum(idx, total_frames - 1)  # 确保索引不超出范围
# #             try:
# #                 props_feats.append(f[idx])
# #             except IndexError:
# #                 # print(f.size(), (s_frame, e_frame))
# #                 exit(0)
# #             # props_len.append(props_feats[-1].size(0))
# #
# #     props_feats = torch.stack(props_feats, 0)
# #
# #     return props_feats, None, None
# #
# #
# # def get_max_score_timestamps(scores, durations):
# #     # 假设scores的维度为[4, 8, 1, 32, 32]
# #     # 初始化输出时间戳列表，确保维度为 (4, 8, 3, 2) 来存储前三个最大值的时间戳
# #     out_max_timestamps = torch.zeros(scores.size(0), scores.size(1), 3, 2)
# #     # 初始化输出分数列表，确保维度为 (4, 8, 3) 来存储前三个最大值的分数
# #     out_max_scores = torch.zeros(scores.size(0), scores.size(1), 3)
# #
# #     for batch_idx, (score_sent, duration) in enumerate(zip(scores, durations)):
# #         for text_idx, score in enumerate(score_sent.squeeze()):  # 去掉维度为1的维度
# #             # 如果当前分数矩阵不全为0
# #             if score.sum() > 1e-3:
# #                 T = score.shape[-1]
# #                 # 将score展平并获取前三个最大值及其索引
# #                 flat_score = score.view(-1)
# #                 top3_values, top3_max_indices = torch.topk(flat_score, 3)  # 获取前三个最大值及其索引
# #
# #                 for i, max_index in enumerate(top3_max_indices):
# #                     max_position = np.unravel_index(max_index.item(), (T, T))
# #                     # 计算时间
# #                     target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
# #                     max_timestamp = (np.array(max_position).astype(float) / target_size * duration)
# #                     # 更新输出时间戳列表
# #                     out_max_timestamps[batch_idx, text_idx, i] = torch.tensor(max_timestamp)
# #                     # 更新输出分数列表
# #                     out_max_scores[batch_idx, text_idx, i] = top3_values[i]
# #
# #     # 返回包含时间戳和分数的元组
# #     return out_max_timestamps, out_max_scores
# #
# # # 注意：需要定义 config.DATASET.NUM_SAMPLE_CLIPS 和 config.DATASET.TARGET_STRIDE 或直接替换为对应的数值
# #
# # def sim_matrix_training(text_embeds, vid_embeds_pooled):
# #     """
# #     Computes the similarity matrix using pooled video frames
# #
# #     Output
# #         sims: num_texts x num_vids
# #     """
# #     text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
# #     vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)
# #
# #
# #     sims = torch.mm(text_embeds, vid_embeds_pooled.t())
# #
# #
# #     # num_texts x embed_dim x num_vids
# #     vid_embeds_pooled = vid_embeds_pooled.permute(1, 2, 0)
# #     # num_texts x 1 x embed_dim
# #     text_embeds = text_embeds.unsqueeze(1)
# #
# #     sims = torch.bmm(text_embeds, vid_embeds_pooled).squeeze(1)
# #
# #     return sims
# #
# #
# #
# #
# # trans=DualTransformer().cuda()
# #
# #
# #
# # def nms(dets, thresh=0.4, top_k=-1):
# #     """Pure Python NMS baseline."""
# #     if len(dets) == 0: return []
# #     order = np.arange(0,len(dets),1)
# #     dets = np.array(dets)
# #     x1 = dets[:, 0]
# #     x2 = dets[:, 1]
# #     lengths = x2 - x1
# #     keep = []
# #     while order.size > 0:
# #         i = order[0]
# #         keep.append(i)
# #         if len(keep) == top_k:
# #             break
# #         xx1 = np.maximum(x1[i], x1[order[1:]])
# #         xx2 = np.minimum(x2[i], x2[order[1:]])
# #         inter = np.maximum(0.0, xx2 - xx1)
# #         ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
# #         inds = np.where(ovr <= thresh)[0]
# #         order = order[inds + 1]
# #
# #     return dets[keep]
# #
# # class DoubleAttentionLayer(nn.Module):
# #     def __init__(self, in_channels, c_m, c_n,k =1 ):
# #         super(DoubleAttentionLayer, self).__init__()
# #
# #         self.K           = k
# #         self.c_m = c_m
# #         self.c_n = c_n
# #         self.softmax     = nn.Softmax()
# #         self.in_channels = in_channels
# #
# #         # self.convA = nn.Conv2d(in_channels, c_m, 1)
# #         # self.convB = nn.Conv2d(in_channels, c_n, 1)
# #         # self.convV = nn.Conv2d(in_channels, c_n, 1)
# #
# #         self.convA = nn.Conv3d(in_channels, c_m, 1)
# #         self.convB = nn.Conv3d(in_channels, c_n, 1)
# #         self.convV = nn.Conv3d(in_channels, c_n, 1)
# #
# #     def forward(self, x):
# #
# #         b, c, d, h, w = x.size()
# #
# #         assert c == self.in_channels,'input channel not equal!'
# #         #assert b//self.K == self.in_channels, 'input channel not equal!'
# #
# #         A = self.convA(x)
# #         B = self.convB(x)
# #         V = self.convV(x)
# #
# #         batch = int(b/self.K)
# #
# #         tmpA = A.view( batch, self.K, self.c_m, d*h*w ).permute(0,2,1,3).view( batch, self.c_m, self.K*d*h*w )
# #         tmpB = B.view( batch, self.K, self.c_n, d*h*w ).permute(0,2,1,3).view( batch*self.c_n, self.K*d*h*w )
# #         tmpV = V.view( batch, self.K, self.c_n, d*h*w ).permute(0,1,3,2).contiguous().view( int(b*d*h*w), self.c_n )
# #
# #         softmaxB = self.softmax(tmpB).view( batch, self.c_n, self.K*d*h*w ).permute( 0, 2, 1)  #batch, self.K*h*w, self.c_n
# #         softmaxV = self.softmax(tmpV).view( batch, self.K*d*h*w, self.c_n ).permute( 0, 2, 1)  #batch, self.c_n  , self.K*h*w
# #
# #         tmpG     = tmpA.matmul( softmaxB )      #batch, self.c_m, self.c_n
# #         tmpZ     = tmpG.matmul( softmaxV ) #batch, self.c_m, self.K*h*w
# #         tmpZ     = tmpZ.view(batch, self.c_m, self.K,d*h*w).permute( 0, 2, 1,3).view( int(b), self.c_m, d, h, w )
# #         return tmpZ
# #
# #
# #
# # class TAN(nn.Module):
# #     def __init__(self):
# #         super(TAN, self).__init__()
# #
# #         self.frame_layer = getattr(frame_modules, config.TAN.FRAME_MODULE.NAME)(config.TAN.FRAME_MODULE.PARAMS)
# #         self.prop_layer = getattr(prop_modules, config.TAN.PROP_MODULE.NAME)(config.TAN.PROP_MODULE.PARAMS)
# #         self.fusion_layer = getattr(fusion_modules, config.TAN.FUSION_MODULE.NAME)(config.TAN.FUSION_MODULE.PARAMS)
# #         self.map_layer = getattr(map_modules, config.TAN.MAP_MODULE.NAME)(config.TAN.MAP_MODULE.PARAMS)
# #         self.pred_layer1 = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)
# #         self.pred_layer2 = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)
# #         self.pred_layer3 = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)
# #         self.bmn_layer = bmn_layer.BMN()
# #         self.nlblock = DoubleAttentionLayer(512, 512, 512)
# #         self.reg_token1 = nn.Embedding(1, 512)
# #         self.reg_token2 = nn.Embedding(1, 512)
# #         self.reg_token3 = nn.Embedding(1, 512)
# #         self.vl_pos_embed1 = nn.Embedding(9, 512)
# #         self.vl_pos_embed2 = nn.Embedding(9, 512)
# #         self.vl_pos_embed3 = nn.Embedding(33, 512)
# #         d_model = 128
# #         self.pos_feat = torch.zeros((d_model*3, 8, 32, 32))
# #         self.vl_transformer1 = build_vl_transformer()
# #         # self.vl_transformer2 = build_vl_transformer()
# #         # self.vl_transformer3 = build_vl_transformer()
# #         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
# #         self.logit_scale2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
# #         self.logit_scale3 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
# #         for k in range(8):
# #             for i in range(32):
# #                 for j in range(i, 32):
# #                     self.pos_feat[0:d_model, k, i, j] = get_positional_encoding(d_model, k+1)
# #                     self.pos_feat[d_model:(d_model*2), k, i, j] = get_positional_encoding(d_model, i+1)
# #                     self.pos_feat[(d_model*2):(d_model*3), k, i, j] = get_positional_encoding(d_model, j+1)
# #         self.pos_feat = self.pos_feat.cuda()
# #         self.word_pos_encoder = SinusoidalPositionalEmbedding(512, 0, 20)
# #         self.mask_vec = nn.Parameter(torch.zeros(512).float(), requires_grad=True)
# #         # self.c_l = nn.Conv1d(in_channels=16,out_channels=512,kernel_size=1)
# #         self.conv1x1=nn.Linear(1024,512)
# #         # self.gg=nn.Linear(1024,512)
# #         # self.trans = DualTransformer()
# #         self.DynamicGCNFusion=DynamicGCNFusion()
# #     # textual_input, textual_mask, sentence_mask, visual_input, duration, weights_list, ids_list
# #     def forward(self, textual_input, textual_mask, sentence_mask, visual_input,duration,weights_list,ids_list):
# #         # visual_input: (b,256, input_size) i.e.(32,256,500)
# #         # textual_input: (b,K,seq,300) tensor
# #         # textual_mask: (b,K,seq,1) tensor
# #         # sentence_mask: (b,K,1) tensor
# #         # print(ids_list)
# #         # print(textual_mask.shape)
# #         # print(weights_list.shape)
# #
# #         aa= textual_mask.size(2)
# #
# #
# #         batch_size = textual_input.size(0)
# #         textual_mask1=textual_mask.squeeze().view(batch_size*8,aa)
# #         seq = textual_input.size(2)
# #         # print(textual_input.shape)torch.Size([4, 8, 24, 300])
# #         # identical as single
# #         vis_h,visual_output = self.frame_layer(visual_input.transpose(1, 2)) #vis_h (b,512,64)
# #
# #         visual_output_g = vis_h
# #         # print(vis_h.shape)torch.Size([16, 512, 32])
# #
# #         #visual_output_g b,512,32
# #         visual_output_g=visual_output_g.permute(2, 0, 1)
# #
# #         video_mask = torch.full((batch_size, 32), False).cuda()
# #
# #         tgt_src_video = self.reg_token3.weight.unsqueeze(1).repeat(1, batch_size, 1)
# #
# #         tgt_mask_t = torch.zeros((batch_size, 1)).to(torch.bool).cuda()
# #
# #         tgt_mask_video=tgt_mask_t
# #
# #         vl_pos_video = self.vl_pos_embed3.weight.unsqueeze(1).repeat(1, batch_size, 1)
# #
# #         vl_src_video = torch.cat([tgt_src_video,visual_output_g], dim=0)
# #
# #         vl_mask_video = torch.cat([tgt_mask_video, video_mask], dim=1)
# #         # print(vl_mask_t.shape)
# #         # print(vl_src_t.shape)
# #
# #         vg_hs_video = self.vl_transformer1(vl_src_video, vl_mask_video, vl_pos_video)  # (1+L+N)xBxC
# #         vis_h = vg_hs_video[1:, :, :].permute(1, 2, 0)
# #         vg_hs_video = vg_hs_video[0]
# #
# #
# #
# #
# #         map_h, map_mask = self.prop_layer(vis_h) #map_h (b,512,64,64) map_mask (b,1,64,64)
# #         # map_h = self.bmn_layer(vis_h)
# #         # print(map_h.shape)torch.Size([16, 512, 32, 32])
# #         # vg_hs_video = map_h[:, :, -1, -1]
# #         map_size = map_h.size(3)
# #         # different due to dense
# #         map_mask_1=np.zeros((batch_size,1,32,32))
# #         for i in range(batch_size):
# #             np.fill_diagonal(map_mask_1[i,0],1)
# #             map_mask_1[i,0] = np.triu(np.ones((32,32)))
# #         map_mask=torch.from_numpy(map_mask_1).cuda().float()
# #         # print(textual_mask.shape)torch.Size([4, 8, 24, 1])
# #         # print(weights_list.shape)torch.Size([4, 8, 24, 1])
# #         # print(map_h.shape)
# #         fused_h, map_mask,txt_h,txt_h_a,_ = self.fusion_layer(textual_input, textual_mask, sentence_mask, map_h, map_mask)
# #         # fused_h (b, K,512,64,64)
# #         # print(txt_h_a.shape)torch.Size([2, 8, 24, 512])
# #         # print(txt_h.shape)([16, 8, 512, 1, 1])
# #         # print(sentence_mask.shape)torch.Size([16, 8, 1])
# # #################################################################################通过图卷积构建不同层级查询文本###############################################
# #         txt_h_b=txt_h.view(batch_size , 8,512)
# #         batch_output_list = []  # 用于存储每次计算后的 x
# #         batch_output_g_list=[]
# #         for i in range(batch_size):
# #             txt_h_b_i=txt_h_b[i,:,:]#维度8，512
# #             sentence_mask_i=sentence_mask[i,:,:]
# #             # print(sentence_mask_i.shape)torch.Size([8, 1])
# #             # 统计有多少个1
# #             a = torch.sum(sentence_mask_i).item()
# #             # 生成 ids 列表
# #             # 将 a 转换为整数
# #             ids = [str(i) for i in range(0, int(a))]
# #
# #             # 将掩码展开为一维并应用掩码，取出掩码为1的特征
# #             masked_features = txt_h_b_i[sentence_mask_i.squeeze(1) == 1]
# #
# #             result = None
# #             while result is None:
# #                 result = generate_group_layers(ids)
# #
# #             group_layers_list = convert_layers_to_list(result)
# #             group_layers_list = [[group[0]] for group in group_layers_list]
# #             # print(result)
# #             # x = masked_features[0,:].unsqueeze(0).cuda()
# #             x = masked_features.cuda()
# #             # print(x.shape)
# #             output_list = []  # 用于存储每次计算后的 x
# #             for index, layer_groups in enumerate(group_layers_list):
# #                 if index == 0:
# #                     x=x
# #                 edge_index = create_edge_index(layer_groups)
# #                 edge_index=edge_index.cuda()
# #                 # print(x.shape)torch.Size([5, 512])
# #                 x = self.DynamicGCNFusion(x, edge_index)
# #                 # print(x.shape)
# #                 output_list.append(x)  # 将计算的 x 添加到列表中，使用 clone() 防止原值改变
# #             # 将 output_list 转换为一个 tensor
# #             output_tensor = torch.stack(output_list)  # 转换为 tensor
# #             # print(output_tensor.shape)
# #             last_vector = output_tensor[0, -1, :]
# #             batch_output_g_list.append(last_vector)
# #
# #             # 获取当前张量的形状
# #             current_shape = output_tensor.shape
# #             # 需要填充的目标大小为 (8, 8, 512)
# #             target_size = (8, 8, current_shape[2])
# #             # 计算需要填充的大小
# #             padding_height = target_size[0] - current_shape[0]  # 第一个维度的填充量
# #             padding_width = target_size[1] - current_shape[1]  # 第二个维度的填充量
# #             # 如果填充量大于0才进行填充
# #             if padding_height > 0 or padding_width > 0:
# #                 padded_tensor = F.pad(output_tensor, (0, 0, 0, padding_width, 0, padding_height), "constant", 0)
# #             else:
# #                 padded_tensor = output_tensor  # 如果已经是 8x8 的张量，直接返回
# #             # print(padded_tensor.shape)  # 输出 (8, 8, 512)
# #             batch_output_list.append(padded_tensor)
# #         batch_output_tensor = torch.stack( batch_output_list)  # 转换为 tensor
# #         vg_hs_t = torch.stack(batch_output_g_list, dim=0)
# #         # print(batch_output_tensor.shape)torch.Size([16, 8, 8, 512])
# #         #########################################################################################################################################################
# #         fused_h = fused_h.view(batch_size * 8, 512, map_size, map_size)  # fused_h (b*8,512,64,64)
# #
# #         map_mask = map_mask.view(batch_size * 8, 1, map_size, map_size)
# #         sentence_mask = sentence_mask.view(batch_size * 8, 1)
# #         sentence_mask = sentence_mask[:, :, None, None]  # sentence_mask (b*8,1, 1, 1)
# #         map_mask = map_mask * sentence_mask
# #
# #         # different due to conv3d
# #         # map_mask (b*8,1,64,64) -> (b,1,8,64,64)
# #         # fused_h  (b*8,512,64,64) -> (b,512,8,64,64)
# #         map_mask = map_mask.view(batch_size, 8, 1, map_size, map_size)
# #         map_mask = map_mask.permute(0, 2, 1, 3, 4)
# #         fused_h = fused_h.view(batch_size, 8, 512, map_size, map_size)
# #         fused_h = fused_h.permute(0, 2, 1, 3, 4)
# #         fused_h = torch.cat((self.pos_feat.repeat(fused_h.size(0), 1, 1, 1, 1).cuda(), fused_h), dim=1)
# #         # embed()
# #         fused_h = self.map_layer(fused_h, map_mask) #fused_h (b,512,8,64,64)
# #         # fused_h = self.nlblock(fused_h) * map_mask
# #         # different due to conv3d
# #         # map_mask (b,1,8,64,64)  -> (b*8,1,64,64)
# #         # fused_h  (b,512,8,64,64) -> (b*8,512,64,64)
# #         fused_h = fused_h.permute(0, 2, 1, 3, 4)
# #         map_mask = map_mask.permute(0, 2, 1, 3, 4)
# #
# #         fused_h = fused_h.contiguous().view(batch_size*8, 512, map_size, map_size)
# #         map_mask = map_mask.contiguous().view(batch_size*8, 1, map_size, map_size)
# #         # print(fused_h.shape)torch.Size([2，8, 512, 32, 32])
# #         prediction = self.pred_layer1(fused_h)  #prediction (b*K,1,64,64)
# #         fused_h_t=fused_h.clone()
# #         prediction = prediction * map_mask #prediction (b*K,1,64,64)
# #         prediction = prediction.view(batch_size, 8, 1, map_size, map_size)
# #         # print(prediction.shape)torch.Size([2, 8, 1, 32, 32])
# #         map_mask = map_mask.view(batch_size, 8, 1, map_size, map_size)
# #         # print(map_mask.shape)torch.Size([2, 8, 1, 32, 32])上三角为1，下三角全是0。
# #         tmp_shape = prediction.shape
# #         joint_prob = torch.sigmoid(prediction) * map_mask
# #         # mask = torch.ones(batch_size, 8, 1, dtype=torch.bool).cuda()
# #
# #         ################################################################################计算最佳的提案所对应的融合特征#####################################################################
# #         # joint_prob = torch.sigmoid(prediction) * map_mask
# #         # print(joint_prob.shape)torch.Size([2, 8, 1, 32, 32])
# #         weight_1, targets_tmp = torch.max(joint_prob.flatten(-2), dim=-1)
# #
# #         # 获取每个位置的前三个最大值及其索引
# #         values, indices = torch.topk(joint_prob.flatten(-2), 3, dim=-1)
# #         mask3 = sentence_mask.view(batch_size, 8).bool().unsqueeze(-1).unsqueeze(-1).expand_as(indices)
# #         indices = (indices * mask3).squeeze(2)
# #
# #         first_elements = indices[:, :, 0]
# #         # 使用unsqueeze增加一个维度，使形状变为[4, 8, 1]
# #         reshaped_tensor_first = first_elements.unsqueeze(-1)
# #
# #         second_elements = indices[:, :, 1]
# #         # 使用unsqueeze增加一个维度，使形状变为[4, 8, 1]
# #         targets_tmp_second = second_elements.unsqueeze(-1)
# #
# #         last_elements = indices[:, :, 2]
# #         # 使用unsqueeze增加一个维度，使形状变为[4, 8, 1]
# #         targets_tmp_last = last_elements.unsqueeze(-1)
# #
# #         # print(targets_tmp)tensor([[[159],[159],[159],[223],[  0], [  0],[  0],[  0]],[[159], [159],[159],[  0], [  0],[  0],[  0],[  0]]], device='cuda:0')
# #         # print(targets_tmp.shape)torch.Size([2, 8, 1])
# #         # print(weight_1.shape)torch.Size([2, 8, 1])
# #         targets = torch.zeros(tmp_shape[0], tmp_shape[1], tmp_shape[-2] * tmp_shape[-1]).cuda()
# #         targets_second = torch.zeros(tmp_shape[0], tmp_shape[1], tmp_shape[-2] * tmp_shape[-1]).cuda()
# #         targets_last = torch.zeros(tmp_shape[0], tmp_shape[1], tmp_shape[-2] * tmp_shape[-1]).cuda()
# #
# #         targets.scatter_(2, targets_tmp, 1)
# #         targets_second.scatter_(2, targets_tmp_second, 1)
# #         targets_last.scatter_(2, targets_tmp_last, 1)
# #
# #         targets = torch.reshape(targets, tmp_shape) * map_mask
# #         targets_second = torch.reshape(targets_second, tmp_shape) * map_mask
# #         targets_last = torch.reshape(targets_last, tmp_shape) * map_mask
# #
# #         # print(targets.shape)torch.Size([4, 8, 1, 32, 32])
# #
# #         non_zero_indices = torch.nonzero(targets)
# #         non_zero_indices_second = torch.nonzero(targets_second)
# #         non_zero_indices_last = torch.nonzero(targets_last)
# #
# #         # 打印非零元素的索引
# #         # print(non_zero_indices)tensor([[ 0,  0,  0,  4, 31],[ 0,  1,  0,  4, 31],[ 0,  2,  0,  4, 31],[ 0,  3,  0,  6, 31],[ 1,  0,  0,  4, 31],[ 1,  1,  0,  4, 31],[ 1,  2,  0,  4, 31]], device='cuda:0')
# #         # print(non_zero_indices[0,:])tensor([ 0,  0,  0,  4, 31], device='cuda:0')
# #         # 调整 fused_h 的形状
# #         fused_h_reshaped = fused_h.view(batch_size, 8, 512, 32, 32)
# #
# #         map_h_c = map_h.unsqueeze(1).repeat(1, 8, 1, 1, 1)
# #         # print(map_h_c.shape)
# #         # 初始化结果容器，尺寸为 [batch_size, 8, 512]
# #         # 其中8代表最大文本数量，512是特征维度
# #         results = torch.zeros(batch_size, 8, 512).cuda()
# #         results_second = torch.zeros(batch_size, 8, 512).cuda()
# #         results_last = torch.zeros(batch_size, 8, 512).cuda()
# #
# #         mask = torch.ones(batch_size, 8, 1, dtype=torch.bool).cuda()
# #
# #         for index in non_zero_indices:
# #             # 提取特定索引的元素
# #             element = fused_h_reshaped[index[0], index[1], :, index[3], index[4]]
# #             # 填充到结果容器中
# #             results[index[0], index[1]] = element
# #             # 更新掩码：已填充的位置设置为 False
# #             mask[index[0], index[1], 0] = False
# #         # print(results.shape)torch.Size([2, 8, 512])
# #         mask_squeezed = torch.squeeze(mask, dim=2)
# #         results = results.view(8, batch_size, 512)
# #
# #         for index in non_zero_indices_second:
# #             # 提取特定索引的元素
# #             element = fused_h_reshaped[index[0], index[1], :, index[3], index[4]]
# #             # 填充到结果容器中
# #             results_second[index[0], index[1]] = element
# #         results_second = results_second.view(8, batch_size, 512)
# #
# #         for index in non_zero_indices_last:
# #             # 提取特定索引的元素
# #             element = fused_h_reshaped[index[0], index[1], :, index[3], index[4]]
# #             # 填充到结果容器中
# #             results_last[index[0], index[1]] = element
# #
# #         results_last = results_last.view(8, batch_size, 512)
# #
# #         results = results * 0.4 + results_second * 0.3 + results_last * 0.3
# #
# #         ############################################################################################################################
# #
# #         tgt_src_v = self.reg_token1.weight.unsqueeze(1).repeat(1, batch_size, 1)
# #         # print(tgt_src_v.shape)
# #         tgt_mask_v = torch.zeros((batch_size, 1)).to(tgt_src_v.cuda()).to(torch.bool)
# #         # print(tgt_mask_v.shape)
# #
# #         # pos
# #         vl_pos_v = self.vl_pos_embed1.weight.unsqueeze(1).repeat(1, batch_size, 1)
# #         # print(vl_pos_v.shape)
# #
# #         # TF
# #         vl_src = torch.cat([tgt_src_v, results.cuda()], dim=0)
# #         vl_mask = torch.cat([tgt_mask_v, mask_squeezed.cuda()], dim=1)
# #         # print(vl_mask.shape)
# #         # print(vl_src.shape)
# #
# #         vg_hs_v_grounding = self.vl_transformer1(vl_src, vl_mask, vl_pos_v)  # (1+L+N)xBxC
# #         vg_hs_v_grounding = vg_hs_v_grounding[0]
# #         # print(vg_hs_v_grounding.shape)torch.Size([16, 512])
# # #######################################################################################################################################################################
# #         # #计算全局文本信息
# #         # # print(txt_h.shape)
# #         # # aaaaa=txt_h.squeeze(-1).squeeze(-1)
# #         # txt_g = txt_h.squeeze(-1).squeeze(-1).permute(1, 0, 2)
# #         # # print(txt_g.shape)
# #         #
# #         #
# #         # mask_t_g=torch.squeeze(mask, dim=2).cuda()
# #         # # txt_g = txt_g.permute(1, 0, 2).masked_fill(mask_t_g.unsqueeze(-1).expand(2, 8, 512), 0).permute(1, 0, 2)
# #         # # 文本target token
# #         # tgt_src_t = self.reg_token2.weight.unsqueeze(1).repeat(1, batch_size, 1)
# #         # # print(tgt_src_t.shape)
# #         # tgt_mask_t = torch.zeros((batch_size, 1)).to(tgt_src_t.device).to(torch.bool).cuda()
# #         # # print(tgt_mask_t.shape)
# #         #
# #         #
# #         # #pos
# #         # vl_pos_t = self.vl_pos_embed2.weight.unsqueeze(1).repeat(1, batch_size, 1)
# #         # # print(vl_pos_t.shape)
# #         #
# #         # #TF
# #         # vl_src_t = torch.cat([tgt_src_t,txt_g], dim=0)
# #         # vl_mask_t = torch.cat([tgt_mask_t, mask_t_g], dim=1)
# #         # # print(vl_mask_t.shape)
# #         # # print(vl_src_t.shape)
# #         #
# #         # vg_hs_t = self.vl_transformer1(vl_src_t, vl_mask_t, vl_pos_t)  # (1+L+N)xBxC
# #         # vg_hs_t = vg_hs_t[0]
# #         # # print(vg_hs_t.shape)
# # ##############################################################################################################
# #         # #visual_output_g b,512,32
# #         # visual_output_g=visual_output_g.permute(2, 0, 1)
# #         #
# #         # video_mask = torch.full((batch_size, 32), False).cuda()
# #         #
# #         # tgt_src_video = self.reg_token3.weight.unsqueeze(1).repeat(1, batch_size, 1)
# #         #
# #         # tgt_mask_video=tgt_mask_t
# #         #
# #         # vl_pos_video = self.vl_pos_embed3.weight.unsqueeze(1).repeat(1, batch_size, 1)
# #         #
# #         # vl_src_video = torch.cat([tgt_src_video,visual_output_g], dim=0)
# #         #
# #         # vl_mask_video = torch.cat([tgt_mask_video, video_mask], dim=1)
# #         # # print(vl_mask_t.shape)
# #         # # print(vl_src_t.shape)
# #         #
# #         # vg_hs_video = self.vl_transformer1(vl_src_video, vl_mask_video, vl_pos_video)  # (1+L+N)xBxC
# #         # vg_hs_video = vg_hs_video[0]
# #
# # ###################################################################################################
# #         # vg_hs_v=visual_output_g+ vg_hs_v
# #         # vg_hs_v=torch.cat((vg_hs_video,vg_hs_v) ,dim=1)
# #         # vg_hs_v=self.conv1x1(vg_hs_v)
# #         # vg_hs_v = vg_hs_v / vg_hs_v.norm(dim=-1, keepdim=True)
# #         vg_hs_t = vg_hs_t / vg_hs_t.norm(dim=-1, keepdim=True)
# #         # vg_hs_video = vg_hs_video / vg_hs_video.norm(dim=-1, keepdim=True)
# #         vg_hs_video=vg_hs_v_grounding/vg_hs_v_grounding.norm(dim=-1, keepdim=True)
# #         # sims= torch.matmul(vg_hs_v,vg_hs_t.T)
# #         sims2 = torch.matmul(vg_hs_video, vg_hs_t.T)
# #         # sims3 = torch.matmul(vg_hs_video, vg_hs_v.T)
# #         # sims=sims1*0.4+sims2*0.6
# #         # print(sims)
# #  ####################################################################################################################################################
# #         # vg_hs_v,vg_hs_t
# #         jj,weight_3=get_max_score_timestamps(joint_prob, duration)
# #         jj=jj.view(batch_size * 8,3,2)
# #
# #         weight_3=weight_3.view(batch_size * 8,3).squeeze(1)
# #
# #         # print(www.shape)
# #         # print(jj.shape)torch.Size([2, 8, 3, 2])
# #         # print(textual_mask.shape)torch.Size([2, 8, 24, 1])
# #         # print(jj.shape)orch.Size([2, 8, 2])
# #         # print(visual_input.shape)torch.Size([2, 256, 500])
# #         # vg_hs_video,vg_hs_t
# #
# #         # print(props_chosen.shape)torch.Size([128, 3, 2])
# #         # print(ori_frames_feat.shape)torch.Size([128, 200, 256])
# #         duration_tensor = torch.tensor(duration).cuda()
# #         # q=duration_tensor.repeat_interleave(8)
# #         # qq=sentence_mask.view(batch_size * 8)
# #         duration_tensors = duration_tensor.repeat_interleave(8) * sentence_mask.view(batch_size * 8)
# #         props_feat, props_len, props_mask = _generate_proposals_feat(visual_input.unsqueeze(1).repeat(1,8,1,1).view(batch_size * 8,256,500), jj,duration_tensors)
# #         # print(props_len)16
# #
# #
# #         # print(props_feat)
# #         # torch.Size([16, 16, 500])
# #         # print(props_feat.shape)torch.Size([48, 16, 500])
# #         # props_feat = props_feat.view(2, 8, 16, 500)
# #         # print(props_feat.shape)
# #
# # #############################################################掩码文本特征#################################################
# #         # print(weights_list.shape)
# #         # print(txt_h_a.shape)
# #         # print(textual_mask.shape)
# #         # torch.Size([2, 8, 24, 1])
# #         # torch.Size([2, 8, 24, 512])
# #         # torch.Size([2, 8, 24, 1])
# #         words_pos = self.word_pos_encoder(txt_h_a.view(batch_size * 8,aa,512))
# #         words_len = textual_mask.sum(dim=2, keepdim=True).view(batch_size * 8)
# #         # print(words_len)tensor([ 7.,  5.,  6.,  5.,  0.,  0.,  0.,  0., 24., 21.,  0.,  0.,  0.,  0.,
# #         #  0.,  0.], device='cuda:0')
# #         # print(words_len.shape)[8]
# #
# #         # 对每个批次中的每个文本权重进行归一化处理
# #         for batch_index in range(weights_list.shape[0]):  # 遍历批次
# #             for text_index in range(weights_list.shape[1]):  # 遍历文本
# #                 # 计算当前文本的权重总和
# #                 weight_sum = weights_list[batch_index, text_index, :, :].sum()
# #
# #                 # 如果权重总和不为零，则进行归一化
# #                 if weight_sum > 0:
# #                     weights_list[batch_index, text_index, :, :] /= weight_sum
# #
# #         words_feat = self._mask_words(txt_h_a.view(batch_size * 8,aa,512), words_len,aa, weights=weights_list.view(batch_size * 8,aa)) + words_pos
# #
# #         words_feat1 = words_feat
# #
# #         ids_list = ids_list.view(batch_size * 8,aa)
# #
# #         num_proposals = 3
# #         bsz = batch_size * 8
# #
# #         words_mask1 = textual_mask1.unsqueeze(1) \
# #             .expand(bsz, num_proposals, -1).contiguous().view(bsz * num_proposals, -1)
# #         # print(words_mask1.shape)torch.Size([64, 24])
# #         ids_list = ids_list.unsqueeze(1) \
# #             .expand(bsz, num_proposals, -1).contiguous().view(bsz * num_proposals, -1)
# #         words_feat1 = words_feat1.unsqueeze(1) \
# #             .expand(bsz, num_proposals, -1, -1).contiguous().view(bsz * num_proposals, words_mask1.size(1), -1)
# #
# #         props_feat1=props_feat
# #         fc= nn.Linear(500,512).cuda()
# #         props_feat1=fc(props_feat1).cuda()
# #         # print(props_feat1.shape)torch.Size([48, 16, 512])
# #
# #         # None
# #         # torch.Size([64, 16, 512])
# #         # torch.Size([64, 24, 512])
# #         # torch.Size([64, 24])
# #
# #         fc_comp = nn.Linear(512, 8001).cuda()
# #
# #         _, h = trans(props_feat1, props_mask, words_feat1, words_mask1, decoding=2)
# #         # print(h.shape)torch.Size([64, 24, 512])
# #         words_logit = fc_comp(h)
# #
# #
# #
# #         weights=None
# #         # print(vg_hs_t.shape)
# #         return prediction, map_mask, self.logit_scale, sims2, self.logit_scale2,  self.logit_scale3, jj, weight_3, words_logit, ids_list, weights, words_mask1, vg_hs_video, vg_hs_t, batch_output_tensor
# #     def _mask_words(self, words_feat, words_len, aa,weights=None):
# #         token = self.mask_vec.cuda().unsqueeze(0).unsqueeze(0)
# #
# #         masked_words = []
# #         for i, l in enumerate(words_len):
# #             if l > 0:  # 跳过长度为0的文本
# #                 l = int(l)
# #                 num_masked_words = l // 3  # 计算要掩码的单词数量
# #                 masked_word = torch.zeros([aa], dtype=torch.uint8).cuda()  # 初始化掩码
# #                 p = weights[i, :l].cpu().numpy()  # 使用weights的实际长度l
# #                 choices = np.random.choice(np.arange(1, l + 1), num_masked_words, replace=False,
# #                                            p=p / p.sum())  # 根据weights随机选择
# #
# #                 masked_word[choices - 1] = 1  # 更新掩码，注意索引调整
# #                 masked_words.append(masked_word)
# #             else:  # 对于长度为0的文本，添加全零掩码
# #                 masked_words.append(torch.zeros([aa], dtype=torch.uint8).cuda())
# #         # 转换masked_words为张量并调整维度以匹配words_feat
# #         masked_words = torch.stack(masked_words, 0).unsqueeze(-1)
# #         masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
# #         masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
# #         words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec
# #         return words_feat1
# #
# #
# #
# #
# # class SinusoidalPositionalEmbedding(nn.Module):
# #     """This module produces sinusoidal positional embeddings of any length.
# #
# #     Padding symbols are ignored.
# #     """
# #
# #     def __init__(self, embedding_dim, padding_idx, init_size=1024):
# #         super().__init__()
# #         self.embedding_dim = embedding_dim
# #         self.padding_idx = padding_idx
# #         self.weights = SinusoidalPositionalEmbedding.get_embedding(
# #             init_size,
# #             embedding_dim,
# #             padding_idx,
# #         )
# #
# #     @staticmethod
# #     def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
# #         """Build sinusoidal embeddings.
# #
# #         This matches the implementation in tensor2tensor, but differs slightly
# #         from the description in Section 3.5 of "Attention Is All You Need".
# #         """
# #         half_dim = embedding_dim // 2
# #         import math
# #         emb = math.log(10000) / (half_dim - 1)
# #         emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
# #         emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
# #         emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
# #         if embedding_dim % 2 == 1:
# #             # zero pad
# #             emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
# #         if padding_idx is not None:
# #             emb[padding_idx, :] = 0
# #         return emb
# #
# #     def forward(self, input, **kwargs):
# #         bsz, seq_len, _ = input.size()
# #         max_pos = seq_len
# #         if self.weights is None or max_pos > self.weights.size(0):
# #             # recompute/expand embeddings if needed
# #             self.weights = SinusoidalPositionalEmbedding.get_embedding(
# #                 max_pos,
# #                 self.embedding_dim,
# #                 self.padding_idx,
# #             )
# #         self.weights = self.weights.cuda(input.device)[:max_pos]
# #         return self.weights.unsqueeze(0)
# #
# #     def max_positions(self):
# #         """Maximum number of supported positions."""
# #         return int(1e5)  # an arbitrary large number
# #
# #
# #
#
#
#
#
#
# from torch import nn
# from lib.core.config import config
# import lib.models.frame_modules as frame_modules
# import lib.models.prop_modules as prop_modules
# import lib.models.map_modules as map_modules
# import lib.models.fusion_modules as fusion_modules
# import lib.models.bmn_modules as bmn_layer
# from IPython import embed
#
# import math
# import numpy as np
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# from .vl_transformer import build_vl_transformer
#
# from lib.models.transformer import DualTransformer
#
#
#
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# import random
# import time
#
# class DynamicGCNFusion(torch.nn.Module):
#     def __init__(self):
#         super(DynamicGCNFusion, self).__init__()
#         self.input_dim = 512  # 直接定义input_dim
#         self.hidden_dim = 512  # 直接定义hidden_dim
#         self.conv1 = GCNConv(self.input_dim, self.hidden_dim)
#         self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
#
#     def forward(self, x, edge_index):
#         # 图卷积层1
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         # 图卷积层2
#         x = self.conv2(x, edge_index)
#         return x
# class DynamicLSTMFusion(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(DynamicLSTMFusion, self).__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
#
#     def forward(self, x):
#         # x should be of shape [batch_size, seq_length, input_dim]
#         output, (hidden, cell) = self.lstm(x)
#         # Return only the last sequence's hidden state
#         return hidden[-1]  # last layer, last timestep hidden state
#
# def generate_group_layers(ids):
#     start_time = time.time()
#     layers = {1: [[id] for id in ids]}
#     total_levels = len(ids)
#
#     for level in range(2, total_levels):
#         while True:
#             current_layer = []
#             all_possible_combinations = set()
#
#             for base_group in layers[level - 1]:
#                 available_ids = [id for id in ids if id not in base_group]
#                 random.shuffle(available_ids)
#                 for chosen_id in available_ids:
#                     new_group = base_group + [chosen_id]
#                     new_group_sorted_tuple = tuple(sorted(new_group))
#                     if new_group_sorted_tuple not in all_possible_combinations:
#                         all_possible_combinations.add(new_group_sorted_tuple)
#                         current_layer.append(new_group)
#                         break
#
#             if len(current_layer) == len(ids):
#                 layers[level] = current_layer
#                 break
#
#             if time.time() - start_time > 0.0001:  # Timeout check
#                 return None
#
#     last_layer = []
#     layer_last_groups = layers[total_levels - 1]
#     for base_group in layer_last_groups:
#         available_ids = [id for id in ids if id not in base_group]
#         random.shuffle(available_ids)
#         for chosen_id in available_ids:
#             new_group = base_group + [chosen_id]
#             new_group_sorted_tuple = tuple(sorted(new_group))
#             if new_group_sorted_tuple not in last_layer:
#                 last_layer.append(new_group)
#                 if len(last_layer) == len(ids):
#                     break
#         if len(last_layer) == len(ids):
#             break
#
#     layers[total_levels] = last_layer
#     return layers
#
# def convert_layers_to_list(layers):
#     return [[list(map(int, group)) for group in layers[level]] for level in sorted(layers)]
#
# def create_sequence_input(vectors, groups):
#     # Concatenate vector representations according to group sequences
#     sequences = [torch.stack([vectors[int(id)] for id in group], dim=0) for group in groups]
#     return torch.stack(sequences, dim=0)
#
# # def generate_group_layers(ids):
# #     start_time = time.time()
# #     layers = {1: [[id] for id in ids]}
# #     total_levels = len(ids)
# #
# #     for level in range(2, total_levels):
# #         while True:
# #             current_layer = []
# #             all_possible_combinations = set()
# #
# #             for base_group in layers[level - 1]:
# #                 available_ids = [id for id in ids if id not in base_group]
# #                 random.shuffle(available_ids)
# #                 for chosen_id in available_ids:
# #                     new_group = base_group + [chosen_id]
# #                     new_group_sorted_tuple = tuple(sorted(new_group))
# #                     if new_group_sorted_tuple not in all_possible_combinations:
# #                         all_possible_combinations.add(new_group_sorted_tuple)
# #                         current_layer.append(new_group)
# #                         break
# #
# #             if len(current_layer) == len(ids):
# #                 layers[level] = current_layer
# #                 break
# #
# #             if time.time() - start_time > 0.0001:  # 改进了超时时间为1秒
# #                 return None
# #
# #     last_layer = []
# #     layer_last_groups = layers[total_levels - 1]
# #     for base_group in layer_last_groups:
# #         available_ids = [id for id in ids if id not in base_group]
# #         random.shuffle(available_ids)
# #         for chosen_id in available_ids:
# #             new_group = base_group + [chosen_id]
# #             new_group_sorted_tuple = tuple(sorted(new_group))
# #             if new_group_sorted_tuple not in last_layer:
# #                 last_layer.append(new_group)
# #                 if len(last_layer) == len(ids):
# #                     break
# #         if len(last_layer) == len(ids):
# #             break
# #
# #     layers[total_levels] = last_layer
# #     return layers
# #
# # def convert_layers_to_list(layers):
# #     return [[list(map(int, group)) for group in layers[level]] for level in sorted(layers)]
# #
# # def create_edge_index(groups):
# #     src, dst = [], []
# #     for group in groups:
# #         for i in range(len(group)):
# #             for j in range(len(group)):
# #                 if i != j:
# #                     src.append(group[i])
# #                     dst.append(group[j])
# #     return torch.tensor([src, dst], dtype=torch.long)
#
# def iou(pred, gt):
#     assert isinstance(pred, list) and isinstance(gt, list)
#     pred_is_list = isinstance(pred[0], list)
#     gt_is_list = isinstance(gt[0], list)
#     if not pred_is_list: pred = [pred]
#     if not gt_is_list: gt = [gt]
#     pred, gt = np.array(pred), np.array(gt)
#     inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
#     inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
#     inter = np.maximum(0.0, inter_right - inter_left)
#     union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
#     union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
#     union = np.maximum(0.0, union_right - union_left)
#     # 避免除以零：在分母上加一个很小的数
#     overlap = 1.0 * inter / (union + 1e-6)
#     if not gt_is_list:
#         overlap = overlap[:, 0]
#     if not pred_is_list:
#         overlap = overlap[0]
#     return overlap
#
#
# def calculate_overlaps(duration_tensor, y_t, num_clips, num_sentences, batch_size):
#     """
#     Calculate overlaps between time segments and given time points.
#
#     Args:
#         duration_tensor (Tensor): Tensor containing durations for each batch.
#         y_t (Tensor): Tensor containing start and end times for sentences.
#         num_clips (int): Number of time segments.
#         num_sentences (int): Number of sentences.
#         batch_size (int): Size of each batch.
#
#     Returns:
#         Tensor: A tensor containing overlap values.
#     """
#
#     overlaps_list = []
#
#     for batch in range(batch_size):
#         overlaps_batch = []
#         duration = duration_tensor[batch].item()  # 获取当前批次的持续时间
#
#         for sentence in range(num_sentences):
#             gt_s_time, gt_e_time = y_t[batch, sentence, 0, 0].item(), y_t[batch, sentence, 0, 1].item()
#
#             if gt_s_time == 0 and gt_e_time == 0:
#                 overlaps_batch.append(torch.zeros(1, num_clips, num_clips))
#                 continue
#
#             s_times = torch.arange(0, num_clips).float() * duration / num_clips
#             e_times = torch.arange(1, num_clips + 1).float() * duration / num_clips
#             overlaps = iou(torch.stack([s_times[:, None].expand(-1, num_clips),
#                                         e_times[None, :].expand(num_clips, -1)], dim=2).view(-1, 2).tolist(),
#                            [gt_s_time, gt_e_time]).reshape(num_clips, num_clips)
#
#             overlaps_batch.append(torch.from_numpy(overlaps).unsqueeze(0))
#
#         overlaps_list.append(torch.cat(overlaps_batch, 0))
#
#     return torch.stack(overlaps_list, 0)
#
#
# def get_max_score_one_timestamps(scores, durations):
#     # 假设scores的维度为[4, 8, 1, 32, 32]
#     # 初始化输出时间戳列表，确保维度为 (4, 8, 3, 2) 来存储前三个最大值的时间戳
#     out_max_timestamps = torch.zeros(scores.size(0), scores.size(1), 1, 2)
#     # 初始化输出分数列表，确保维度为 (4, 8, 3) 来存储前三个最大值的分数
#     out_max_scores = torch.zeros(scores.size(0), scores.size(1), 1)
#
#     for batch_idx, (score_sent, duration) in enumerate(zip(scores, durations)):
#         for text_idx, score in enumerate(score_sent.squeeze()):  # 去掉维度为1的维度
#             # 如果当前分数矩阵不全为0
#             if score.sum() > 1e-3:
#                 T = score.shape[-1]
#                 # 将score展平并获取前三个最大值及其索引
#                 flat_score = score.view(-1)
#                 top1_values, top1_max_indices = torch.topk(flat_score, 1)  # 获取前三个最大值及其索引
#
#                 for i, max_index in enumerate(top1_max_indices):
#                     max_position = np.unravel_index(max_index.item(), (T, T))
#                     # 计算时间
#                     target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
#                     max_timestamp = (np.array(max_position).astype(float) / target_size * duration)
#                     # 更新输出时间戳列表
#                     out_max_timestamps[batch_idx, text_idx, i] = torch.tensor(max_timestamp)
#                     # 更新输出分数列表
#                     out_max_scores[batch_idx, text_idx, i] = top1_values[i]
#
#     # 返回包含时间戳和分数的元组
#     return out_max_timestamps, out_max_scores
#
#
# def get_positional_encoding(d_model, idx):
#     positional_encoding = torch.zeros((d_model,))  # (max_length, d_model)
#     i = idx
#     for j in range(d_model):
#         if j % 2 == 0:
#             positional_encoding[j] = math.sin(i / math.pow(10000, j / d_model))
#         else:
#             positional_encoding[j] = math.cos(i / math.pow(10000, (j - 1) / d_model))
#
# #     positional_encoding = positional_encoding.unsqueeze(0)  # (1, max_length, d_model)
#
#     return positional_encoding
#
# def get_proposal_results(scores, durations):
#         # assume all valid scores are larger than one
#         out_sorted_times = []
#         for score_sent, duration in zip(scores, durations):
#             sent_times = []
#             for score in score_sent:
#                 if score.sum() < 1e-3:
#                     break
#                 T = score.shape[-1]
#                 sorted_indexs = np.dstack(
#                     np.unravel_index(np.argsort(score.cpu().detach().numpy().ravel())[::-1], (T, T))).tolist()
#                 sorted_indexs = np.array([item for item in sorted_indexs[0] if item[0] <= item[1]]).astype(float)
#
#                 sorted_indexs[:, 1] = sorted_indexs[:, 1] + 1
#                 sorted_indexs = torch.from_numpy(sorted_indexs).cuda()
#                 target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
#                 sent_times.append((sorted_indexs.float() / target_size * duration).tolist())
#             out_sorted_times.append(sent_times)
#         return out_sorted_times
#
# #
# def _generate_proposals_feat(frames_feat, props, duration):
#     props_feats = []
#     # props_len = []
#
#     for f, p, d in zip(frames_feat, props, duration):
#         total_frames = f.shape[0]
#
#         for s, e in p:
#             if s == 0 and e == 0 and d==0:
#                 # 如果 s 和 e 都是 0，则随机初始化特征
#                 # random_feat = torch.normal(0,0.1,size=(16,512)).cuda()
#                 random_feat = torch.zeros(16, 500).cuda()
#                 props_feats.append(random_feat)
#                 # props_len.append(batch_size * 8)
#                 continue
#
#             # 将 s 和 e 映射到相对于视频持续时间的比例，并转换为帧索引
#             s_frame = int((s / d) * total_frames)
#             e_frame = int((e / d) * total_frames)
#             s_frame, e_frame = min(s_frame, total_frames - 1), min(e_frame, total_frames - 1)
#
#             # 生成索引并提取特征
#             idx = np.linspace(start=s_frame, stop=e_frame, num=16, endpoint=False).astype(np.int32)
#             idx = np.minimum(idx, total_frames - 1)  # 确保索引不超出范围
#             try:
#                 props_feats.append(f[idx])
#             except IndexError:
#                 # print(f.size(), (s_frame, e_frame))
#                 exit(0)
#             # props_len.append(props_feats[-1].size(0))
#
#     props_feats = torch.stack(props_feats, 0)
#
#     return props_feats, None, None
#
#
# def get_max_score_timestamps(scores, durations):
#     # 假设scores的维度为[4, 8, 1, 32, 32]
#     # 初始化输出时间戳列表，确保维度为 (4, 8, 3, 2) 来存储前三个最大值的时间戳
#     out_max_timestamps = torch.zeros(scores.size(0), scores.size(1), 3, 2)
#     # 初始化输出分数列表，确保维度为 (4, 8, 3) 来存储前三个最大值的分数
#     out_max_scores = torch.zeros(scores.size(0), scores.size(1), 3)
#
#     for batch_idx, (score_sent, duration) in enumerate(zip(scores, durations)):
#         for text_idx, score in enumerate(score_sent.squeeze()):  # 去掉维度为1的维度
#             # 如果当前分数矩阵不全为0
#             if score.sum() > 1e-3:
#                 T = score.shape[-1]
#                 # 将score展平并获取前三个最大值及其索引
#                 flat_score = score.view(-1)
#                 top3_values, top3_max_indices = torch.topk(flat_score, 3)  # 获取前三个最大值及其索引
#
#                 for i, max_index in enumerate(top3_max_indices):
#                     max_position = np.unravel_index(max_index.item(), (T, T))
#                     # 计算时间
#                     target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
#                     max_timestamp = (np.array(max_position).astype(float) / target_size * duration)
#                     # 更新输出时间戳列表
#                     out_max_timestamps[batch_idx, text_idx, i] = torch.tensor(max_timestamp)
#                     # 更新输出分数列表
#                     out_max_scores[batch_idx, text_idx, i] = top3_values[i]
#
#     # 返回包含时间戳和分数的元组
#     return out_max_timestamps, out_max_scores
#
# # 注意：需要定义 config.DATASET.NUM_SAMPLE_CLIPS 和 config.DATASET.TARGET_STRIDE 或直接替换为对应的数值
#
# def sim_matrix_training(text_embeds, vid_embeds_pooled):
#     """
#     Computes the similarity matrix using pooled video frames
#
#     Output
#         sims: num_texts x num_vids
#     """
#     text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
#     vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)
#
#
#     sims = torch.mm(text_embeds, vid_embeds_pooled.t())
#
#
#     # num_texts x embed_dim x num_vids
#     vid_embeds_pooled = vid_embeds_pooled.permute(1, 2, 0)
#     # num_texts x 1 x embed_dim
#     text_embeds = text_embeds.unsqueeze(1)
#
#     sims = torch.bmm(text_embeds, vid_embeds_pooled).squeeze(1)
#
#     return sims
#
#
#
#
# trans=DualTransformer().cuda()
#
#
#
# def nms(dets, thresh=0.4, top_k=-1):
#     """Pure Python NMS baseline."""
#     if len(dets) == 0: return []
#     order = np.arange(0,len(dets),1)
#     dets = np.array(dets)
#     x1 = dets[:, 0]
#     x2 = dets[:, 1]
#     lengths = x2 - x1
#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         if len(keep) == top_k:
#             break
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         inter = np.maximum(0.0, xx2 - xx1)
#         ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
#         inds = np.where(ovr <= thresh)[0]
#         order = order[inds + 1]
#
#     return dets[keep]
#
# class DoubleAttentionLayer(nn.Module):
#     def __init__(self, in_channels, c_m, c_n,k =1 ):
#         super(DoubleAttentionLayer, self).__init__()
#
#         self.K           = k
#         self.c_m = c_m
#         self.c_n = c_n
#         self.softmax     = nn.Softmax()
#         self.in_channels = in_channels
#
#         # self.convA = nn.Conv2d(in_channels, c_m, 1)
#         # self.convB = nn.Conv2d(in_channels, c_n, 1)
#         # self.convV = nn.Conv2d(in_channels, c_n, 1)
#
#         self.convA = nn.Conv3d(in_channels, c_m, 1)
#         self.convB = nn.Conv3d(in_channels, c_n, 1)
#         self.convV = nn.Conv3d(in_channels, c_n, 1)
#
#     def forward(self, x):
#
#         b, c, d, h, w = x.size()
#
#         assert c == self.in_channels,'input channel not equal!'
#         #assert b//self.K == self.in_channels, 'input channel not equal!'
#
#         A = self.convA(x)
#         B = self.convB(x)
#         V = self.convV(x)
#
#         batch = int(b/self.K)
#
#         tmpA = A.view( batch, self.K, self.c_m, d*h*w ).permute(0,2,1,3).view( batch, self.c_m, self.K*d*h*w )
#         tmpB = B.view( batch, self.K, self.c_n, d*h*w ).permute(0,2,1,3).view( batch*self.c_n, self.K*d*h*w )
#         tmpV = V.view( batch, self.K, self.c_n, d*h*w ).permute(0,1,3,2).contiguous().view( int(b*d*h*w), self.c_n )
#
#         softmaxB = self.softmax(tmpB).view( batch, self.c_n, self.K*d*h*w ).permute( 0, 2, 1)  #batch, self.K*h*w, self.c_n
#         softmaxV = self.softmax(tmpV).view( batch, self.K*d*h*w, self.c_n ).permute( 0, 2, 1)  #batch, self.c_n  , self.K*h*w
#
#         tmpG     = tmpA.matmul( softmaxB )      #batch, self.c_m, self.c_n
#         tmpZ     = tmpG.matmul( softmaxV ) #batch, self.c_m, self.K*h*w
#         tmpZ     = tmpZ.view(batch, self.c_m, self.K,d*h*w).permute( 0, 2, 1,3).view( int(b), self.c_m, d, h, w )
#         return tmpZ
#
#
#
# class TAN(nn.Module):
#     def __init__(self):
#         super(TAN, self).__init__()
#
#         self.frame_layer = getattr(frame_modules, config.TAN.FRAME_MODULE.NAME)(config.TAN.FRAME_MODULE.PARAMS)
#         self.prop_layer = getattr(prop_modules, config.TAN.PROP_MODULE.NAME)(config.TAN.PROP_MODULE.PARAMS)
#         self.fusion_layer = getattr(fusion_modules, config.TAN.FUSION_MODULE.NAME)(config.TAN.FUSION_MODULE.PARAMS)
#         self.map_layer = getattr(map_modules, config.TAN.MAP_MODULE.NAME)(config.TAN.MAP_MODULE.PARAMS)
#         self.pred_layer1 = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)
#         self.pred_layer2 = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)
#         self.pred_layer3 = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)
#         self.bmn_layer = bmn_layer.BMN()
#         self.nlblock = DoubleAttentionLayer(512, 512, 512)
#         self.reg_token1 = nn.Embedding(1, 512)
#         self.reg_token2 = nn.Embedding(1, 512)
#         self.reg_token3 = nn.Embedding(1, 512)
#         self.vl_pos_embed1 = nn.Embedding(9, 512)
#         self.vl_pos_embed2 = nn.Embedding(9, 512)
#         self.vl_pos_embed3 = nn.Embedding(33, 512)
#         d_model = 128
#         self.pos_feat = torch.zeros((d_model*3, 8, 32, 32))
#         self.vl_transformer1 = build_vl_transformer()
#         # self.vl_transformer2 = build_vl_transformer()
#         # self.vl_transformer3 = build_vl_transformer()
#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
#         self.logit_scale2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
#         self.logit_scale3 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
#         for k in range(8):
#             for i in range(32):
#                 for j in range(i, 32):
#                     self.pos_feat[0:d_model, k, i, j] = get_positional_encoding(d_model, k+1)
#                     self.pos_feat[d_model:(d_model*2), k, i, j] = get_positional_encoding(d_model, i+1)
#                     self.pos_feat[(d_model*2):(d_model*3), k, i, j] = get_positional_encoding(d_model, j+1)
#         self.pos_feat = self.pos_feat.cuda()
#         self.word_pos_encoder = SinusoidalPositionalEmbedding(512, 0, 20)
#         self.mask_vec = nn.Parameter(torch.zeros(512).float(), requires_grad=True)
#         # self.c_l = nn.Conv1d(in_channels=16,out_channels=512,kernel_size=1)
#         self.conv1x1=nn.Linear(1024,512)
#         # self.gg=nn.Linear(1024,512)
#         # self.trans = DualTransformer()
#         self.DynamicGCNFusion=DynamicGCNFusion()
#         self.DynamicLSTMFusion=DynamicLSTMFusion(512,512)
#         self.fc=nn.Linear(512,512)
#     # textual_input, textual_mask, sentence_mask, visual_input, duration, weights_list, ids_list
#     def forward(self, textual_input, textual_mask, sentence_mask, visual_input,duration,weights_list,ids_list):
#         # visual_input: (b,256, input_size) i.e.(32,256,500)
#         # textual_input: (b,K,seq,300) tensor
#         # textual_mask: (b,K,seq,1) tensor
#         # sentence_mask: (b,K,1) tensor
#         # print(ids_list)
#         # print(textual_mask.shape)
#         # print(weights_list.shape)
#
#         aa= textual_mask.size(2)
#
#
#         batch_size = textual_input.size(0)
#         textual_mask1=textual_mask.squeeze().view(batch_size*8,aa)
#         seq = textual_input.size(2)
#         # print(textual_input.shape)torch.Size([4, 8, 24, 300])
#         # identical as single
#         vis_h,visual_output = self.frame_layer(visual_input.transpose(1, 2)) #vis_h (b,512,64)
#
#         visual_output_g = vis_h
#         # print(vis_h.shape)torch.Size([16, 512, 32])
#
#         #visual_output_g b,512,32
#         visual_output_g=visual_output_g.permute(2, 0, 1)
#
#         video_mask = torch.full((batch_size, 32), False).cuda()
#
#         tgt_src_video = self.reg_token3.weight.unsqueeze(1).repeat(1, batch_size, 1)
#
#         tgt_mask_t = torch.zeros((batch_size, 1)).to(torch.bool).cuda()
#
#         tgt_mask_video=tgt_mask_t
#
#         vl_pos_video = self.vl_pos_embed3.weight.unsqueeze(1).repeat(1, batch_size, 1)
#
#         vl_src_video = torch.cat([tgt_src_video,visual_output_g], dim=0)
#
#         vl_mask_video = torch.cat([tgt_mask_video, video_mask], dim=1)
#         # print(vl_mask_t.shape)
#         # print(vl_src_t.shape)
#
#         vg_hs_video = self.vl_transformer1(vl_src_video, vl_mask_video, vl_pos_video)  # (1+L+N)xBxC
#         vis_h = vg_hs_video[1:, :, :].permute(1, 2, 0)
#         vg_hs_video = vg_hs_video[0]
#
#
#
#
#         map_h, map_mask = self.prop_layer(vis_h) #map_h (b,512,64,64) map_mask (b,1,64,64)
#         # map_h = self.bmn_layer(vis_h)
#         # print(map_h.shape)torch.Size([16, 512, 32, 32])
#         # vg_hs_video = map_h[:, :, -1, -1]
#         map_size = map_h.size(3)
#         # different due to dense
#         map_mask_1=np.zeros((batch_size,1,32,32))
#         for i in range(batch_size):
#             np.fill_diagonal(map_mask_1[i,0],1)
#             map_mask_1[i,0] = np.triu(np.ones((32,32)))
#         map_mask=torch.from_numpy(map_mask_1).cuda().float()
#         # print(textual_mask.shape)torch.Size([4, 8, 24, 1])
#         # print(weights_list.shape)torch.Size([4, 8, 24, 1])
#         # print(map_h.shape)
#         fused_h, map_mask,txt_h,txt_h_a,_,batch_output_tensor,vg_hs_t = self.fusion_layer(textual_input, textual_mask, sentence_mask, map_h, map_mask)
#         # fused_h (b, K,512,64,64)
#         # print(txt_h_a.shape)torch.Size([2, 8, 24, 512])
#         # print(txt_h.shape)([16, 8, 512, 1, 1])
#         # print(sentence_mask.shape)torch.Size([16, 8, 1])
# #################################################################################通过图卷积构建不同层级查询文本###############################################
#         # txt_h_b=txt_h.view(batch_size , 8,512)
#         # batch_output_list = []  # 用于存储每次计算后的 x
#         # batch_output_g_list=[]
#         # for i in range(batch_size):
#         #     txt_h_b_i=txt_h_b[i,:,:]#维度8，512
#         #     sentence_mask_i=sentence_mask[i,:,:]
#         #     # print(sentence_mask_i.shape)torch.Size([8, 1])
#         #     # 统计有多少个1
#         #     a = torch.sum(sentence_mask_i).item()
#         #     # 生成 ids 列表
#         #     # 将 a 转换为整数
#         #     ids = [str(i) for i in range(0, int(a))]
#         #
#         #     # 将掩码展开为一维并应用掩码，取出掩码为1的特征
#         #     masked_features = txt_h_b_i[sentence_mask_i.squeeze(1) == 1]
#         #
#         #     result = None
#         #     while result is None:
#         #         result = generate_group_layers(ids)
#         #
#         #     group_layers_list = convert_layers_to_list(result)
#         #     # group_layers_list = [[group[0]] for group in group_layers_list]
#         #     # print(result)
#         #     # x = masked_features[0,:].unsqueeze(0).cuda()
#         #     x = masked_features.cuda()
#         #     # print(x.shape)
#         #     output_list = []  # 用于存储每次计算后的 x
#         #     for index, layer_groups in enumerate(group_layers_list):
#         #         if index == 0:
#         #             x=x
#         #         # edge_index = create_edge_index(layer_groups)
#         #         # edge_index = layer_groups
#         #         # edge_index=edge_index.cuda()
#         #         sequence_input = create_sequence_input(x, layer_groups)
#         #         # print(x.shape)torch.Size([5, 512])
#         #         x = self.DynamicLSTMFusion(sequence_input)
#         #         x=self.fc(x)
#         #         # print(x.shape)
#         #         output_list.append(x)  # 将计算的 x 添加到列表中，使用 clone() 防止原值改变
#         #     # 将 output_list 转换为一个 tensor
#         #     output_tensor = torch.stack(output_list)  # 转换为 tensor
#         #     # print(output_tensor.shape)
#         #     last_vector = output_tensor[0, -1, :]
#         #     batch_output_g_list.append(last_vector)
#         #
#         #     # 获取当前张量的形状
#         #     current_shape = output_tensor.shape
#         #     # 需要填充的目标大小为 (8, 8, 512)
#         #     target_size = (8, 8, current_shape[2])
#         #     # 计算需要填充的大小
#         #     padding_height = target_size[0] - current_shape[0]  # 第一个维度的填充量
#         #     padding_width = target_size[1] - current_shape[1]  # 第二个维度的填充量
#         #     # 如果填充量大于0才进行填充
#         #     if padding_height > 0 or padding_width > 0:
#         #         padded_tensor = F.pad(output_tensor, (0, 0, 0, padding_width, 0, padding_height), "constant", 0)
#         #     else:
#         #         padded_tensor = output_tensor  # 如果已经是 8x8 的张量，直接返回
#         #     # print(padded_tensor.shape)  # 输出 (8, 8, 512)
#         #     batch_output_list.append(padded_tensor)
#         # batch_output_tensor = torch.stack( batch_output_list)  # 转换为 tensor
#         # # print(batch_output_tensor.shape)
#         # vg_hs_t = torch.stack(batch_output_g_list, dim=0)
#         # # print(batch_output_tensor.shape)torch.Size([16, 8, 8, 512])
#         #########################################################################################################################################################
#         fused_h = fused_h.view(batch_size * 8, 512, map_size, map_size)  # fused_h (b*8,512,64,64)
#
#         map_mask = map_mask.view(batch_size * 8, 1, map_size, map_size)
#         sentence_mask = sentence_mask.view(batch_size * 8, 1)
#         sentence_mask = sentence_mask[:, :, None, None]  # sentence_mask (b*8,1, 1, 1)
#         map_mask = map_mask * sentence_mask
#
#         # different due to conv3d
#         # map_mask (b*8,1,64,64) -> (b,1,8,64,64)
#         # fused_h  (b*8,512,64,64) -> (b,512,8,64,64)
#         map_mask = map_mask.view(batch_size, 8, 1, map_size, map_size)
#         map_mask = map_mask.permute(0, 2, 1, 3, 4)
#         fused_h = fused_h.view(batch_size, 8, 512, map_size, map_size)
#         fused_h = fused_h.permute(0, 2, 1, 3, 4)
#         fused_h = torch.cat((self.pos_feat.repeat(fused_h.size(0), 1, 1, 1, 1).cuda(), fused_h), dim=1)
#         # embed()
#         fused_h = self.map_layer(fused_h, map_mask) #fused_h (b,512,8,64,64)
#         # fused_h = self.nlblock(fused_h) * map_mask
#         # different due to conv3d
#         # map_mask (b,1,8,64,64)  -> (b*8,1,64,64)
#         # fused_h  (b,512,8,64,64) -> (b*8,512,64,64)
#         fused_h = fused_h.permute(0, 2, 1, 3, 4)
#         map_mask = map_mask.permute(0, 2, 1, 3, 4)
#
#         fused_h = fused_h.contiguous().view(batch_size*8, 512, map_size, map_size)
#         map_mask = map_mask.contiguous().view(batch_size*8, 1, map_size, map_size)
#         # print(fused_h.shape)torch.Size([2，8, 512, 32, 32])
#         prediction = self.pred_layer1(fused_h)  #prediction (b*K,1,64,64)
#         fused_h_t=fused_h.clone()
#         prediction = prediction * map_mask #prediction (b*K,1,64,64)
#         prediction = prediction.view(batch_size, 8, 1, map_size, map_size)
#         # print(prediction.shape)torch.Size([2, 8, 1, 32, 32])
#         map_mask = map_mask.view(batch_size, 8, 1, map_size, map_size)
#         # print(map_mask.shape)torch.Size([2, 8, 1, 32, 32])上三角为1，下三角全是0。
#         tmp_shape = prediction.shape
#         joint_prob = torch.sigmoid(prediction) * map_mask
#         # mask = torch.ones(batch_size, 8, 1, dtype=torch.bool).cuda()
#
# #         ################################################################################计算最佳的提案所对应的融合特征#####################################################################
# #         # joint_prob = torch.sigmoid(prediction) * map_mask
# #         # print(joint_prob.shape)torch.Size([2, 8, 1, 32, 32])
# #         weight_1, targets_tmp = torch.max(joint_prob.flatten(-2), dim=-1)
# #
# #         # 获取每个位置的前三个最大值及其索引
# #         values, indices = torch.topk(joint_prob.flatten(-2), 3, dim=-1)
# #         mask3 = sentence_mask.view(batch_size, 8).bool().unsqueeze(-1).unsqueeze(-1).expand_as(indices)
# #         indices = (indices * mask3).squeeze(2)
# #
# #         first_elements = indices[:, :, 0]
# #         # 使用unsqueeze增加一个维度，使形状变为[4, 8, 1]
# #         reshaped_tensor_first = first_elements.unsqueeze(-1)
# #
# #         second_elements = indices[:, :, 1]
# #         # 使用unsqueeze增加一个维度，使形状变为[4, 8, 1]
# #         targets_tmp_second = second_elements.unsqueeze(-1)
# #
# #         last_elements = indices[:, :, 2]
# #         # 使用unsqueeze增加一个维度，使形状变为[4, 8, 1]
# #         targets_tmp_last = last_elements.unsqueeze(-1)
# #
# #         # print(targets_tmp)tensor([[[159],[159],[159],[223],[  0], [  0],[  0],[  0]],[[159], [159],[159],[  0], [  0],[  0],[  0],[  0]]], device='cuda:0')
# #         # print(targets_tmp.shape)torch.Size([2, 8, 1])
# #         # print(weight_1.shape)torch.Size([2, 8, 1])
# #         targets = torch.zeros(tmp_shape[0], tmp_shape[1], tmp_shape[-2] * tmp_shape[-1]).cuda()
# #         targets_second = torch.zeros(tmp_shape[0], tmp_shape[1], tmp_shape[-2] * tmp_shape[-1]).cuda()
# #         targets_last = torch.zeros(tmp_shape[0], tmp_shape[1], tmp_shape[-2] * tmp_shape[-1]).cuda()
# #
# #         targets.scatter_(2, targets_tmp, 1)
# #         targets_second.scatter_(2, targets_tmp_second, 1)
# #         targets_last.scatter_(2, targets_tmp_last, 1)
# #
# #         targets = torch.reshape(targets, tmp_shape) * map_mask
# #         targets_second = torch.reshape(targets_second, tmp_shape) * map_mask
# #         targets_last = torch.reshape(targets_last, tmp_shape) * map_mask
# #
# #         # print(targets.shape)torch.Size([4, 8, 1, 32, 32])
# #
# #         non_zero_indices = torch.nonzero(targets)
# #         non_zero_indices_second = torch.nonzero(targets_second)
# #         non_zero_indices_last = torch.nonzero(targets_last)
# #
# #         # 打印非零元素的索引
# #         # print(non_zero_indices)tensor([[ 0,  0,  0,  4, 31],[ 0,  1,  0,  4, 31],[ 0,  2,  0,  4, 31],[ 0,  3,  0,  6, 31],[ 1,  0,  0,  4, 31],[ 1,  1,  0,  4, 31],[ 1,  2,  0,  4, 31]], device='cuda:0')
# #         # print(non_zero_indices[0,:])tensor([ 0,  0,  0,  4, 31], device='cuda:0')
# #         # 调整 fused_h 的形状
# #         fused_h_reshaped = fused_h.view(batch_size, 8, 512, 32, 32)
# #
# #         map_h_c = map_h.unsqueeze(1).repeat(1, 8, 1, 1, 1)
# #         # print(map_h_c.shape)
# #         # 初始化结果容器，尺寸为 [batch_size, 8, 512]
# #         # 其中8代表最大文本数量，512是特征维度
# #         results = torch.zeros(batch_size, 8, 512).cuda()
# #         results_second = torch.zeros(batch_size, 8, 512).cuda()
# #         results_last = torch.zeros(batch_size, 8, 512).cuda()
# #
# #         mask = torch.ones(batch_size, 8, 1, dtype=torch.bool).cuda()
# #
# #         for index in non_zero_indices:
# #             # 提取特定索引的元素
# #             element = fused_h_reshaped[index[0], index[1], :, index[3], index[4]]
# #             # 填充到结果容器中
# #             results[index[0], index[1]] = element
# #             # 更新掩码：已填充的位置设置为 False
# #             mask[index[0], index[1], 0] = False
# #         # print(results.shape)torch.Size([2, 8, 512])
# #         mask_squeezed = torch.squeeze(mask, dim=2)
# #         results = results.view(8, batch_size, 512)
# #
# #         for index in non_zero_indices_second:
# #             # 提取特定索引的元素
# #             element = fused_h_reshaped[index[0], index[1], :, index[3], index[4]]
# #             # 填充到结果容器中
# #             results_second[index[0], index[1]] = element
# #         results_second = results_second.view(8, batch_size, 512)
# #
# #         for index in non_zero_indices_last:
# #             # 提取特定索引的元素
# #             element = fused_h_reshaped[index[0], index[1], :, index[3], index[4]]
# #             # 填充到结果容器中
# #             results_last[index[0], index[1]] = element
# #
# #         results_last = results_last.view(8, batch_size, 512)
# #
# #         results = results * 0.4 + results_second * 0.3 + results_last * 0.3
# #
# #         ############################################################################################################################
# #
# #         tgt_src_v = self.reg_token1.weight.unsqueeze(1).repeat(1, batch_size, 1)
# #         # print(tgt_src_v.shape)
# #         tgt_mask_v = torch.zeros((batch_size, 1)).to(tgt_src_v.cuda()).to(torch.bool)
# #         # print(tgt_mask_v.shape)
# #
# #         # pos
# #         vl_pos_v = self.vl_pos_embed1.weight.unsqueeze(1).repeat(1, batch_size, 1)
# #         # print(vl_pos_v.shape)
# #
# #         # TF
# #         vl_src = torch.cat([tgt_src_v, results.cuda()], dim=0)
# #         vl_mask = torch.cat([tgt_mask_v, mask_squeezed.cuda()], dim=1)
# #         # print(vl_mask.shape)
# #         # print(vl_src.shape)
# #
# #         vg_hs_v_grounding = self.vl_transformer1(vl_src, vl_mask, vl_pos_v)  # (1+L+N)xBxC
# #         vg_hs_v_grounding = vg_hs_v_grounding[0]
# #         # print(vg_hs_v_grounding.shape)torch.Size([16, 512])
# # #######################################################################################################################################################################
#         # #计算全局文本信息
#         # # print(txt_h.shape)
#         # # aaaaa=txt_h.squeeze(-1).squeeze(-1)
#         # txt_g = txt_h.squeeze(-1).squeeze(-1).permute(1, 0, 2)
#         # # print(txt_g.shape)
#         #
#         #
#         # mask_t_g=torch.squeeze(mask, dim=2).cuda()
#         # # txt_g = txt_g.permute(1, 0, 2).masked_fill(mask_t_g.unsqueeze(-1).expand(2, 8, 512), 0).permute(1, 0, 2)
#         # # 文本target token
#         # tgt_src_t = self.reg_token2.weight.unsqueeze(1).repeat(1, batch_size, 1)
#         # # print(tgt_src_t.shape)
#         # tgt_mask_t = torch.zeros((batch_size, 1)).to(tgt_src_t.device).to(torch.bool).cuda()
#         # # print(tgt_mask_t.shape)
#         #
#         #
#         # #pos
#         # vl_pos_t = self.vl_pos_embed2.weight.unsqueeze(1).repeat(1, batch_size, 1)
#         # # print(vl_pos_t.shape)
#         #
#         # #TF
#         # vl_src_t = torch.cat([tgt_src_t,txt_g], dim=0)
#         # vl_mask_t = torch.cat([tgt_mask_t, mask_t_g], dim=1)
#         # # print(vl_mask_t.shape)
#         # # print(vl_src_t.shape)
#         #
#         # vg_hs_t = self.vl_transformer1(vl_src_t, vl_mask_t, vl_pos_t)  # (1+L+N)xBxC
#         # vg_hs_t = vg_hs_t[0]
#         # # print(vg_hs_t.shape)
# ##############################################################################################################
#         # #visual_output_g b,512,32
#         # visual_output_g=visual_output_g.permute(2, 0, 1)
#         #
#         # video_mask = torch.full((batch_size, 32), False).cuda()
#         #
#         # tgt_src_video = self.reg_token3.weight.unsqueeze(1).repeat(1, batch_size, 1)
#         #
#         # tgt_mask_video=tgt_mask_t
#         #
#         # vl_pos_video = self.vl_pos_embed3.weight.unsqueeze(1).repeat(1, batch_size, 1)
#         #
#         # vl_src_video = torch.cat([tgt_src_video,visual_output_g], dim=0)
#         #
#         # vl_mask_video = torch.cat([tgt_mask_video, video_mask], dim=1)
#         # # print(vl_mask_t.shape)
#         # # print(vl_src_t.shape)
#         #
#         # vg_hs_video = self.vl_transformer1(vl_src_video, vl_mask_video, vl_pos_video)  # (1+L+N)xBxC
#         # vg_hs_video = vg_hs_video[0]
#
# ###################################################################################################
#         # vg_hs_v=visual_output_g+ vg_hs_v
#         # vg_hs_v=torch.cat((vg_hs_video,vg_hs_v) ,dim=1)
#         # vg_hs_v=self.conv1x1(vg_hs_v)
#         # vg_hs_v = vg_hs_v / vg_hs_v.norm(dim=-1, keepdim=True)
#         # vg_hs_t = vg_hs_t / vg_hs_t.norm(dim=-1, keepdim=True)
#         # vg_hs_video = vg_hs_video / vg_hs_video.norm(dim=-1, keepdim=True)
#         # vg_hs_video=vg_hs_v_grounding/vg_hs_v_grounding.norm(dim=-1, keepdim=True)
#         # sims= torch.matmul(vg_hs_v,vg_hs_t.T)
#         # sims2 = torch.matmul(vg_hs_video, vg_hs_t.T)
#         sims2=None
#         # sims3 = torch.matmul(vg_hs_video, vg_hs_v.T)
#         # sims=sims1*0.4+sims2*0.6
#         # print(sims)
#  ####################################################################################################################################################
#         # vg_hs_v,vg_hs_t
#         jj,weight_3=get_max_score_timestamps(joint_prob, duration)
#         jj=jj.view(batch_size * 8,3,2)
#
#         weight_3=weight_3.view(batch_size * 8,3).squeeze(1)
#
#         # print(www.shape)
#         # print(jj.shape)torch.Size([2, 8, 3, 2])
#         # print(textual_mask.shape)torch.Size([2, 8, 24, 1])
#         # print(jj.shape)orch.Size([2, 8, 2])
#         # print(visual_input.shape)torch.Size([2, 256, 500])
#         # vg_hs_video,vg_hs_t
#
#         # print(props_chosen.shape)torch.Size([128, 3, 2])
#         # print(ori_frames_feat.shape)torch.Size([128, 200, 256])
#         duration_tensor = torch.tensor(duration).cuda()
#         # q=duration_tensor.repeat_interleave(8)
#         # qq=sentence_mask.view(batch_size * 8)
#         duration_tensors = duration_tensor.repeat_interleave(8) * sentence_mask.view(batch_size * 8)
#         props_feat, props_len, props_mask = _generate_proposals_feat(visual_input.unsqueeze(1).repeat(1,8,1,1).view(batch_size * 8,256,500), jj,duration_tensors)
#         # print(props_len)16
#
#
#         # print(props_feat)
#         # torch.Size([16, 16, 500])
#         # print(props_feat.shape)torch.Size([48, 16, 500])
#         # props_feat = props_feat.view(2, 8, 16, 500)
#         # print(props_feat.shape)
#
# #############################################################掩码文本特征#################################################
#         # print(weights_list.shape)
#         # print(txt_h_a.shape)
#         # print(textual_mask.shape)
#         # torch.Size([2, 8, 24, 1])
#         # torch.Size([2, 8, 24, 512])
#         # torch.Size([2, 8, 24, 1])
#         words_pos = self.word_pos_encoder(txt_h_a.view(batch_size * 8,aa,512))
#         words_len = textual_mask.sum(dim=2, keepdim=True).view(batch_size * 8)
#         # print(words_len)tensor([ 7.,  5.,  6.,  5.,  0.,  0.,  0.,  0., 24., 21.,  0.,  0.,  0.,  0.,
#         #  0.,  0.], device='cuda:0')
#         # print(words_len.shape)[8]
#
#         # 对每个批次中的每个文本权重进行归一化处理
#         for batch_index in range(weights_list.shape[0]):  # 遍历批次
#             for text_index in range(weights_list.shape[1]):  # 遍历文本
#                 # 计算当前文本的权重总和
#                 weight_sum = weights_list[batch_index, text_index, :, :].sum()
#
#                 # 如果权重总和不为零，则进行归一化
#                 if weight_sum > 0:
#                     weights_list[batch_index, text_index, :, :] /= weight_sum
#
#         words_feat = self._mask_words(txt_h_a.view(batch_size * 8,aa,512), words_len,aa, weights=weights_list.view(batch_size * 8,aa)) + words_pos
#
#         words_feat1 = words_feat
#
#         ids_list = ids_list.view(batch_size * 8,aa)
#
#         num_proposals = 3
#         bsz = batch_size * 8
#
#         words_mask1 = textual_mask1.unsqueeze(1) \
#             .expand(bsz, num_proposals, -1).contiguous().view(bsz * num_proposals, -1)
#         # print(words_mask1.shape)torch.Size([64, 24])
#         ids_list = ids_list.unsqueeze(1) \
#             .expand(bsz, num_proposals, -1).contiguous().view(bsz * num_proposals, -1)
#         words_feat1 = words_feat1.unsqueeze(1) \
#             .expand(bsz, num_proposals, -1, -1).contiguous().view(bsz * num_proposals, words_mask1.size(1), -1)
#
#         props_feat1=props_feat
#         fc= nn.Linear(500,512).cuda()
#         props_feat1=fc(props_feat1).cuda()
#         # print(props_feat1.shape)torch.Size([48, 16, 512])
#
#         # None
#         # torch.Size([64, 16, 512])
#         # torch.Size([64, 24, 512])
#         # torch.Size([64, 24])
#
#         fc_comp = nn.Linear(512, 8001).cuda()
#
#         _, h = trans(props_feat1, props_mask, words_feat1, words_mask1, decoding=2)
#         # print(h.shape)torch.Size([64, 24, 512])
#         words_logit = fc_comp(h)
#
#
#
#         weights=None
#         # print(vg_hs_t.shape)
#         return prediction, map_mask, self.logit_scale, sims2, self.logit_scale2,  self.logit_scale3, jj, weight_3, words_logit, ids_list, weights, words_mask1, vg_hs_video, vg_hs_t, batch_output_tensor
#     def _mask_words(self, words_feat, words_len, aa,weights=None):
#         token = self.mask_vec.cuda().unsqueeze(0).unsqueeze(0)
#
#         masked_words = []
#         for i, l in enumerate(words_len):
#             if l > 0:  # 跳过长度为0的文本
#                 l = int(l)
#                 num_masked_words = l // 3  # 计算要掩码的单词数量
#                 masked_word = torch.zeros([aa], dtype=torch.uint8).cuda()  # 初始化掩码
#                 p = weights[i, :l].cpu().numpy()  # 使用weights的实际长度l
#                 choices = np.random.choice(np.arange(1, l + 1), num_masked_words, replace=False,
#                                            p=p / p.sum())  # 根据weights随机选择
#
#                 masked_word[choices - 1] = 1  # 更新掩码，注意索引调整
#                 masked_words.append(masked_word)
#             else:  # 对于长度为0的文本，添加全零掩码
#                 masked_words.append(torch.zeros([aa], dtype=torch.uint8).cuda())
#         # 转换masked_words为张量并调整维度以匹配words_feat
#         masked_words = torch.stack(masked_words, 0).unsqueeze(-1)
#         masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
#         masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
#         words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec
#         return words_feat1
#
#
#
#
# class SinusoidalPositionalEmbedding(nn.Module):
#     """This module produces sinusoidal positional embeddings of any length.
#
#     Padding symbols are ignored.
#     """
#
#     def __init__(self, embedding_dim, padding_idx, init_size=1024):
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.padding_idx = padding_idx
#         self.weights = SinusoidalPositionalEmbedding.get_embedding(
#             init_size,
#             embedding_dim,
#             padding_idx,
#         )
#
#     @staticmethod
#     def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
#         """Build sinusoidal embeddings.
#
#         This matches the implementation in tensor2tensor, but differs slightly
#         from the description in Section 3.5 of "Attention Is All You Need".
#         """
#         half_dim = embedding_dim // 2
#         import math
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
#         emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
#         emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
#         if embedding_dim % 2 == 1:
#             # zero pad
#             emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
#         if padding_idx is not None:
#             emb[padding_idx, :] = 0
#         return emb
#
#     def forward(self, input, **kwargs):
#         bsz, seq_len, _ = input.size()
#         max_pos = seq_len
#         if self.weights is None or max_pos > self.weights.size(0):
#             # recompute/expand embeddings if needed
#             self.weights = SinusoidalPositionalEmbedding.get_embedding(
#                 max_pos,
#                 self.embedding_dim,
#                 self.padding_idx,
#             )
#         self.weights = self.weights.cuda(input.device)[:max_pos]
#         return self.weights.unsqueeze(0)
#
#     def max_positions(self):
#         """Maximum number of supported positions."""
#         return int(1e5)  # an arbitrary large number
#
#
#
from torch import nn
from lib.core.config import config
import lib.models.frame_modules as frame_modules
import lib.models.prop_modules as prop_modules
import lib.models.map_modules as map_modules
import lib.models.fusion_modules as fusion_modules
import lib.models.bmn_modules as bmn_layer
from IPython import embed

import math
import numpy as np
import torch
import torch.nn as nn
# import matplotlib.pyplot as plt
from .vl_transformer import build_vl_transformer

from lib.models.transformer import DualTransformer


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import random
import time




class DynamicGCNFusion(torch.nn.Module):
    def __init__(self):
        super(DynamicGCNFusion, self).__init__()
        self.input_dim = 512  # 直接定义input_dim
        self.hidden_dim = 512  # 直接定义hidden_dim
        self.conv1 = GCNConv(self.input_dim, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)

    def forward(self, x, edge_index):
        # 图卷积层1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 图卷积层2
        x = self.conv2(x, edge_index)
        return x
class DynamicLSTMFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DynamicLSTMFusion, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        # x should be of shape [batch_size, seq_length, input_dim]
        output, (hidden, cell) = self.lstm(x)
        # Return only the last sequence's hidden state
        return hidden[-1]  # last layer, last timestep hidden state

def generate_group_layers(ids):
    start_time = time.time()
    layers = {1: [[id] for id in ids]}
    total_levels = len(ids)

    for level in range(2, total_levels):
        while True:
            current_layer = []
            all_possible_combinations = set()

            for base_group in layers[level - 1]:
                available_ids = [id for id in ids if id not in base_group]
                random.shuffle(available_ids)
                for chosen_id in available_ids:
                    new_group = base_group + [chosen_id]
                    new_group_sorted_tuple = tuple(sorted(new_group))
                    if new_group_sorted_tuple not in all_possible_combinations:
                        all_possible_combinations.add(new_group_sorted_tuple)
                        current_layer.append(new_group)
                        break

            if len(current_layer) == len(ids):
                layers[level] = current_layer
                break

            if time.time() - start_time > 0.0001:  # Timeout check
                return None

    last_layer = []
    layer_last_groups = layers[total_levels - 1]
    for base_group in layer_last_groups:
        available_ids = [id for id in ids if id not in base_group]
        random.shuffle(available_ids)
        for chosen_id in available_ids:
            new_group = base_group + [chosen_id]
            new_group_sorted_tuple = tuple(sorted(new_group))
            if new_group_sorted_tuple not in last_layer:
                last_layer.append(new_group)
                if len(last_layer) == len(ids):
                    break
        if len(last_layer) == len(ids):
            break

    layers[total_levels] = last_layer
    return layers

def convert_layers_to_list(layers):
    return [[list(map(int, group)) for group in layers[level]] for level in sorted(layers)]

def create_sequence_input(vectors, groups):
    # Concatenate vector representations according to group sequences
    sequences = [torch.stack([vectors[int(id)] for id in group], dim=0) for group in groups]
    return torch.stack(sequences, dim=0)

def generate_group_layers(ids):
    start_time = time.time()
    layers = {1: [[id] for id in ids]}
    total_levels = len(ids)

    for level in range(2, total_levels):
        while True:
            current_layer = []
            all_possible_combinations = set()

            for base_group in layers[level - 1]:
                available_ids = [id for id in ids if id not in base_group]
                random.shuffle(available_ids)
                for chosen_id in available_ids:
                    new_group = base_group + [chosen_id]
                    new_group_sorted_tuple = tuple(sorted(new_group))
                    if new_group_sorted_tuple not in all_possible_combinations:
                        all_possible_combinations.add(new_group_sorted_tuple)
                        current_layer.append(new_group)
                        break

            if len(current_layer) == len(ids):
                layers[level] = current_layer
                break

            if time.time() - start_time > 0.0001:  # 改进了超时时间为1秒
                return None

    last_layer = []
    layer_last_groups = layers[total_levels - 1]
    for base_group in layer_last_groups:
        available_ids = [id for id in ids if id not in base_group]
        random.shuffle(available_ids)
        for chosen_id in available_ids:
            new_group = base_group + [chosen_id]
            new_group_sorted_tuple = tuple(sorted(new_group))
            if new_group_sorted_tuple not in last_layer:
                last_layer.append(new_group)
                if len(last_layer) == len(ids):
                    break
        if len(last_layer) == len(ids):
            break

    layers[total_levels] = last_layer
    return layers

def convert_layers_to_list(layers):
    return [[list(map(int, group)) for group in layers[level]] for level in sorted(layers)]

def create_edge_index(groups):
    src, dst = [], []
    for group in groups:
        for i in range(len(group)):
            for j in range(len(group)):
                if i != j:
                    src.append(group[i])
                    dst.append(group[j])
    return torch.tensor([src, dst], dtype=torch.long)
















def iou(pred, gt):
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    if not pred_is_list: pred = [pred]
    if not gt_is_list: gt = [gt]
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(0.0, union_right - union_left)
    # 避免除以零：在分母上加一个很小的数
    overlap = 1.0 * inter / (union + 1e-6)
    if not gt_is_list:
        overlap = overlap[:, 0]
    if not pred_is_list:
        overlap = overlap[0]
    return overlap


def calculate_overlaps(duration_tensor, y_t, num_clips, num_sentences, batch_size):
    """
    Calculate overlaps between time segments and given time points.

    Args:
        duration_tensor (Tensor): Tensor containing durations for each batch.
        y_t (Tensor): Tensor containing start and end times for sentences.
        num_clips (int): Number of time segments.
        num_sentences (int): Number of sentences.
        batch_size (int): Size of each batch.

    Returns:
        Tensor: A tensor containing overlap values.
    """

    overlaps_list = []

    for batch in range(batch_size):
        overlaps_batch = []
        duration = duration_tensor[batch].item()  # 获取当前批次的持续时间

        for sentence in range(num_sentences):
            gt_s_time, gt_e_time = y_t[batch, sentence, 0, 0].item(), y_t[batch, sentence, 0, 1].item()

            if gt_s_time == 0 and gt_e_time == 0:
                overlaps_batch.append(torch.zeros(1, num_clips, num_clips))
                continue

            s_times = torch.arange(0, num_clips).float() * duration / num_clips
            e_times = torch.arange(1, num_clips + 1).float() * duration / num_clips
            overlaps = iou(torch.stack([s_times[:, None].expand(-1, num_clips),
                                        e_times[None, :].expand(num_clips, -1)], dim=2).view(-1, 2).tolist(),
                           [gt_s_time, gt_e_time]).reshape(num_clips, num_clips)

            overlaps_batch.append(torch.from_numpy(overlaps).unsqueeze(0))

        overlaps_list.append(torch.cat(overlaps_batch, 0))

    return torch.stack(overlaps_list, 0)


def get_max_score_one_timestamps(scores, durations):
    # 假设scores的维度为[4, 8, 1, 32, 32]
    # 初始化输出时间戳列表，确保维度为 (4, 8, 3, 2) 来存储前三个最大值的时间戳
    out_max_timestamps = torch.zeros(scores.size(0), scores.size(1), 1, 2)
    # 初始化输出分数列表，确保维度为 (4, 8, 3) 来存储前三个最大值的分数
    out_max_scores = torch.zeros(scores.size(0), scores.size(1), 1)

    for batch_idx, (score_sent, duration) in enumerate(zip(scores, durations)):
        for text_idx, score in enumerate(score_sent.squeeze()):  # 去掉维度为1的维度
            # 如果当前分数矩阵不全为0
            if score.sum() > 1e-3:
                T = score.shape[-1]
                # 将score展平并获取前三个最大值及其索引
                flat_score = score.view(-1)
                top1_values, top1_max_indices = torch.topk(flat_score, 1)  # 获取前三个最大值及其索引

                for i, max_index in enumerate(top1_max_indices):
                    max_position = np.unravel_index(max_index.item(), (T, T))
                    # 计算时间
                    target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
                    max_timestamp = (np.array(max_position).astype(float) / target_size * duration)
                    # 更新输出时间戳列表
                    out_max_timestamps[batch_idx, text_idx, i] = torch.tensor(max_timestamp)
                    # 更新输出分数列表
                    out_max_scores[batch_idx, text_idx, i] = top1_values[i]

    # 返回包含时间戳和分数的元组
    return out_max_timestamps, out_max_scores


def get_positional_encoding(d_model, idx):
    positional_encoding = torch.zeros((d_model,))  # (max_length, d_model)
    i = idx
    for j in range(d_model):
        if j % 2 == 0:
            positional_encoding[j] = math.sin(i / math.pow(10000, j / d_model))
        else:
            positional_encoding[j] = math.cos(i / math.pow(10000, (j - 1) / d_model))

#     positional_encoding = positional_encoding.unsqueeze(0)  # (1, max_length, d_model)

    return positional_encoding

def get_proposal_results(scores, durations):
        # assume all valid scores are larger than one
        out_sorted_times = []
        for score_sent, duration in zip(scores, durations):
            sent_times = []
            for score in score_sent:
                if score.sum() < 1e-3:
                    break
                T = score.shape[-1]
                sorted_indexs = np.dstack(
                    np.unravel_index(np.argsort(score.cpu().detach().numpy().ravel())[::-1], (T, T))).tolist()
                sorted_indexs = np.array([item for item in sorted_indexs[0] if item[0] <= item[1]]).astype(float)

                sorted_indexs[:, 1] = sorted_indexs[:, 1] + 1
                sorted_indexs = torch.from_numpy(sorted_indexs).cuda()
                target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
                sent_times.append((sorted_indexs.float() / target_size * duration).tolist())
            out_sorted_times.append(sent_times)
        return out_sorted_times

#
def _generate_proposals_feat(frames_feat, props, duration):
    props_feats = []
    # props_len = []

    for f, p, d in zip(frames_feat, props, duration):
        total_frames = f.shape[0]

        for s, e in p:
            if s == 0 and e == 0 and d==0:
                # 如果 s 和 e 都是 0，则随机初始化特征
                # random_feat = torch.normal(0,0.1,size=(16,512)).cuda()
                random_feat = torch.zeros(16, 500).cuda()
                props_feats.append(random_feat)
                # props_len.append(batch_size * 8)
                continue

            # 将 s 和 e 映射到相对于视频持续时间的比例，并转换为帧索引
            s_frame = int((s / d) * total_frames)
            e_frame = int((e / d) * total_frames)
            s_frame, e_frame = min(s_frame, total_frames - 1), min(e_frame, total_frames - 1)

            # 生成索引并提取特征
            idx = np.linspace(start=s_frame, stop=e_frame, num=16, endpoint=False).astype(np.int32)
            idx = np.minimum(idx, total_frames - 1)  # 确保索引不超出范围
            try:
                props_feats.append(f[idx])
            except IndexError:
                # print(f.size(), (s_frame, e_frame))
                exit(0)
            # props_len.append(props_feats[-1].size(0))

    props_feats = torch.stack(props_feats, 0)

    return props_feats, None, None

#
#
# def extract_layers_and_indices(score):
#     # 确认输入是二维的tensor
#     assert len(score.shape) == 2, "输入必须是二维的矩阵"
#     rows, cols = score.shape
#
#     # 获取最大值和对应的一维索引
#     max_value = score.max()
#     max_idx = score.argmax()
#
#     # 计算最大值的行列位置
#     row = max_idx // cols
#     col = max_idx % cols
#
#     # 定义辅助函数，安全获取tensor中的值和一维索引
#     def get_value_and_index(x, y):
#         if 0 <= x < rows and 0 <= y < cols:
#             return score[x, y].item(), x * cols + y  # 转换为一维索引
#         else:
#             return None  # 如果索引超出范围，返回None
#
#     # 获取第一层的值（最大值本身）
#     first_layer = [get_value_and_index(row, col)]
#
#     # 获取第二层的值
#     second_layer = []
#     for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#         result = get_value_and_index(row + dx, col + dy)
#         if result:
#             second_layer.append(result)
#
#     # 获取第三层的值
#     third_layer = []
#     for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
#         result = get_value_and_index(row + dx, col + dy)
#         if result:
#             third_layer.append(result)
#
#     # 补全第二层和第三层不足四个元素的情况
#     while len(second_layer) < 4:
#         second_layer.append(second_layer[-1])  # 复制最后一个元素
#
#     while len(third_layer) < 4:
#         third_layer.append(third_layer[-1])  # 复制最后一个元素
#
#     # 将第一层、第二层、第三层合并，获取top 9的值和索引
#     all_layers = first_layer + second_layer[:4] + third_layer[:4]
#
#     # 提取值和索引
#     top9_values = torch.tensor([x[0] for x in all_layers])
#     top9_max_indices = torch.tensor([x[1] for x in all_layers])
#
#     return top9_values, top9_max_indices

#
# def extract_layers_and_indices(score):
#     # 确认输入是二维的tensor
#     assert len(score.shape) == 2, "输入必须是二维的矩阵"
#     rows, cols = score.shape
#
#     # 获取最大值和对应的一维索引
#     max_idx = score.argmax()
#     row = max_idx // cols
#     col = max_idx % cols
#
#     # 定义辅助函数，安全获取tensor中的值和一维索引
#     def get_value_and_index(x, y):
#         if 0 <= x < rows and 0 <= y < cols:
#             return score[x, y].item(), x * cols + y  # 转换为一维索引
#         else:
#             return None  # 如果索引超出范围，返回None
#
#     # 获取第一层的值（最大值本身）
#     first_layer = [get_value_and_index(row, col)]
#
#     # 获取第二层的8个相邻元素
#     second_layer_candidates = []
#     for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
#         result = get_value_and_index(row + dx, col + dy)
#         if result:  # 确保结果不为None
#             second_layer_candidates.append(result)
#
#     # 从第二层的8个候选中选择3个最大的值
#     second_layer = sorted(second_layer_candidates, key=lambda x: x[0], reverse=True)[:3]
#
#     # 获取第三层的12个相邻元素，包围第二层
#     third_layer_candidates = []
#     for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2), (-2, -1), (-2, 1), (2, -1), (2, 1),
#                    (-1, -2), (-1, 2), (1, -2), (1, 2)]:
#         result = get_value_and_index(row + dx, col + dy)
#         if result:  # 确保结果不为None
#             third_layer_candidates.append(result)
#
#     # 从第三层的12个候选中选择3个最大的值
#     third_layer = sorted(third_layer_candidates, key=lambda x: x[0], reverse=True)[:3]
#
#     # 将第一层、第二层、第三层合并，获取top 7的值和索引（1 + 3 + 3 = 7）
#     all_layers = first_layer + second_layer + third_layer
#
#     # 提取值和索引
#     top7_values = torch.tensor([x[0] for x in all_layers])
#     top7_max_indices = torch.tensor([x[1] for x in all_layers])
#
#     return top7_values, top7_max_indices
# import torch
# import numpy as np


def extract_layers_and_indices(score, second_layer_count=1, third_layer_count=1):
    """
    提取最大值及其邻近区域的索引，并允许调节第二层和第三层最大值的数量。

    参数：
    score: 2D tensor，表示得分矩阵。
    second_layer_count: 第二层中要选择的最大值个数，默认值为3。
    third_layer_count: 第三层中要选择的最大值个数，默认值为3。

    返回：
    top_values: 前几层中选择的最大值的张量。
    top_indices: 前几层中最大值对应的一维索引的张量。
    """
    # 确认输入是二维的tensor
    assert len(score.shape) == 2, "输入必须是二维的矩阵"
    rows, cols = score.shape

    # 获取最大值和对应的一维索引
    max_idx = score.argmax()
    row = max_idx // cols
    col = max_idx % cols

    # 定义辅助函数，安全获取tensor中的值和一维索引
    def get_value_and_index(x, y):
        if 0 <= x < rows and 0 <= y < cols:
            return score[x, y].item(), x * cols + y  # 转换为一维索引
        else:
            return None  # 如果索引超出范围，返回None

    # 获取第一层的值（最大值本身）
    first_layer = [get_value_and_index(row, col)]

    # 获取第二层的8个相邻元素
    second_layer_candidates = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        result = get_value_and_index(row + dx, col + dy)
        if result:  # 确保结果不为None
            second_layer_candidates.append(result)

    # 从第二层的8个候选中选择指定数量的最大值
    second_layer = sorted(second_layer_candidates, key=lambda x: x[0], reverse=True)[:second_layer_count]

    # 获取第三层的16个相邻元素，包围第二层
    third_layer_candidates = []
    for dx, dy in [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
                   (2, -2), (2, -1), (2, 0), (2, 1), (2, 2),
                   (-1, -2), (0, -2), (1, -2), (-1, 2), (0, 2), (1, 2)]:
        result = get_value_and_index(row + dx, col + dy)
        if result:  # 确保结果不为None
            third_layer_candidates.append(result)

    # 从第三层的16个候选中选择指定数量的最大值
    third_layer = sorted(third_layer_candidates, key=lambda x: x[0], reverse=True)[:third_layer_count]

    # 将第一层、第二层、第三层合并
    all_layers = first_layer + second_layer + third_layer

    # 提取值和索引
    top_values = torch.tensor([x[0] for x in all_layers])
    top_indices = torch.tensor([x[1] for x in all_layers])

    return top_values, top_indices


def get_max_score_timestamps_9(scores, durations):
    # 假设scores的维度为[4, 8, 1, 32, 32]
    # 初始化输出时间戳列表，确保维度为 (4, 8, 3, 2) 来存储前三个最大值的时间戳
    out_max_timestamps = torch.zeros(scores.size(0), scores.size(1), 3, 2)
    # 初始化输出分数列表，确保维度为 (4, 8, 3) 来存储前三个最大值的分数
    out_max_scores = torch.zeros(scores.size(0), scores.size(1), 3)
    for batch_idx, (score_sent, duration) in enumerate(zip(scores, durations)):
        for text_idx, score in enumerate(score_sent.squeeze()):  # 去掉维度为1的维度
            # 如果当前分数矩阵不全为0
            if score.sum() > 1e-3:
                T = score.shape[-1]

                top9_values, top9_max_indices=extract_layers_and_indices(score)
                for i, max_index in enumerate(top9_max_indices):
                    max_position = np.unravel_index(max_index.item(), (T, T))
                    # 计算时间
                    target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
                    max_timestamp = (np.array(max_position).astype(float) / target_size * duration)
                    # 更新输出时间戳列表
                    out_max_timestamps[batch_idx, text_idx, i] = torch.tensor(max_timestamp)
                    # 更新输出分数列表
                    out_max_scores[batch_idx, text_idx, i] = top9_values[i]
    # 返回包含时间戳和分数的元组
    return out_max_timestamps, out_max_scores

















def get_max_score_timestamps(scores, durations):
    # 假设scores的维度为[4, 8, 1, 32, 32]
    # 初始化输出时间戳列表，确保维度为 (4, 8, 3, 2) 来存储前三个最大值的时间戳
    out_max_timestamps = torch.zeros(scores.size(0), scores.size(1), 3, 2)
    # 初始化输出分数列表，确保维度为 (4, 8, 3) 来存储前三个最大值的分数
    out_max_scores = torch.zeros(scores.size(0), scores.size(1), 3)

    for batch_idx, (score_sent, duration) in enumerate(zip(scores, durations)):
        for text_idx, score in enumerate(score_sent.squeeze()):  # 去掉维度为1的维度
            # 如果当前分数矩阵不全为0
            if score.sum() > 1e-3:
                T = score.shape[-1]
                # 将score展平并获取前三个最大值及其索引
                flat_score = score.view(-1)
                top3_values, top3_max_indices = torch.topk(flat_score, 3)  # 获取前三个最大值及其索引

                for i, max_index in enumerate(top3_max_indices):
                    max_position = np.unravel_index(max_index.item(), (T, T))
                    # 计算时间
                    target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
                    max_timestamp = (np.array(max_position).astype(float) / target_size * duration)
                    # 更新输出时间戳列表
                    out_max_timestamps[batch_idx, text_idx, i] = torch.tensor(max_timestamp)
                    # 更新输出分数列表
                    out_max_scores[batch_idx, text_idx, i] = top3_values[i]

    # 返回包含时间戳和分数的元组
    return out_max_timestamps, out_max_scores

# 注意：需要定义 config.DATASET.NUM_SAMPLE_CLIPS 和 config.DATASET.TARGET_STRIDE 或直接替换为对应的数值

def sim_matrix_training(text_embeds, vid_embeds_pooled):
    """
    Computes the similarity matrix using pooled video frames

    Output
        sims: num_texts x num_vids
    """
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)


    sims = torch.mm(text_embeds, vid_embeds_pooled.t())


    # num_texts x embed_dim x num_vids
    vid_embeds_pooled = vid_embeds_pooled.permute(1, 2, 0)
    # num_texts x 1 x embed_dim
    text_embeds = text_embeds.unsqueeze(1)

    sims = torch.bmm(text_embeds, vid_embeds_pooled).squeeze(1)

    return sims




trans=DualTransformer().cuda()



def nms(dets, thresh=0.4, top_k=-1):
    """Pure Python NMS baseline."""
    if len(dets) == 0: return []
    order = np.arange(0,len(dets),1)
    dets = np.array(dets)
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    lengths = x2 - x1
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(keep) == top_k:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep]

class DoubleAttentionLayer(nn.Module):
    def __init__(self, in_channels, c_m, c_n,k =1 ):
        super(DoubleAttentionLayer, self).__init__()

        self.K           = k
        self.c_m = c_m
        self.c_n = c_n
        self.softmax     = nn.Softmax()
        self.in_channels = in_channels

        # self.convA = nn.Conv2d(in_channels, c_m, 1)
        # self.convB = nn.Conv2d(in_channels, c_n, 1)
        # self.convV = nn.Conv2d(in_channels, c_n, 1)

        self.convA = nn.Conv3d(in_channels, c_m, 1)
        self.convB = nn.Conv3d(in_channels, c_n, 1)
        self.convV = nn.Conv3d(in_channels, c_n, 1)

    def forward(self, x):

        b, c, d, h, w = x.size()

        assert c == self.in_channels,'input channel not equal!'
        #assert b//self.K == self.in_channels, 'input channel not equal!'

        A = self.convA(x)
        B = self.convB(x)
        V = self.convV(x)

        batch = int(b/self.K)

        tmpA = A.view( batch, self.K, self.c_m, d*h*w ).permute(0,2,1,3).view( batch, self.c_m, self.K*d*h*w )
        tmpB = B.view( batch, self.K, self.c_n, d*h*w ).permute(0,2,1,3).view( batch*self.c_n, self.K*d*h*w )
        tmpV = V.view( batch, self.K, self.c_n, d*h*w ).permute(0,1,3,2).contiguous().view( int(b*d*h*w), self.c_n )

        softmaxB = self.softmax(tmpB).view( batch, self.c_n, self.K*d*h*w ).permute( 0, 2, 1)  #batch, self.K*h*w, self.c_n
        softmaxV = self.softmax(tmpV).view( batch, self.K*d*h*w, self.c_n ).permute( 0, 2, 1)  #batch, self.c_n  , self.K*h*w

        tmpG     = tmpA.matmul( softmaxB )      #batch, self.c_m, self.c_n
        tmpZ     = tmpG.matmul( softmaxV ) #batch, self.c_m, self.K*h*w
        tmpZ     = tmpZ.view(batch, self.c_m, self.K,d*h*w).permute( 0, 2, 1,3).view( int(b), self.c_m, d, h, w )
        return tmpZ



class TAN(nn.Module):
    def __init__(self):
        super(TAN, self).__init__()

        self.frame_layer = getattr(frame_modules, config.TAN.FRAME_MODULE.NAME)(config.TAN.FRAME_MODULE.PARAMS)
        self.prop_layer = getattr(prop_modules, config.TAN.PROP_MODULE.NAME)(config.TAN.PROP_MODULE.PARAMS)
        self.fusion_layer = getattr(fusion_modules, config.TAN.FUSION_MODULE.NAME)(config.TAN.FUSION_MODULE.PARAMS)
        self.map_layer = getattr(map_modules, config.TAN.MAP_MODULE.NAME)(config.TAN.MAP_MODULE.PARAMS)
        self.pred_layer1 = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)
        self.pred_layer2 = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)
        self.pred_layer3 = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)
        self.bmn_layer = bmn_layer.BMN()
        self.nlblock = DoubleAttentionLayer(512, 512, 512)
        self.reg_token1 = nn.Embedding(1, 512)
        self.reg_token2 = nn.Embedding(1, 512)
        self.reg_token3 = nn.Embedding(1, 512)
        self.vl_pos_embed1 = nn.Embedding(9, 512)
        self.vl_pos_embed2 = nn.Embedding(9, 512)
        self.vl_pos_embed3 = nn.Embedding(257, 512)
        d_model = 128
        self.pos_feat = torch.zeros((d_model*3, 8, 32, 32))
        self.vl_transformer1 = build_vl_transformer()
        # self.vl_transformer2 = build_vl_transformer()
        # self.vl_transformer3 = build_vl_transformer()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale3 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        for k in range(8):
            for i in range(32):
                for j in range(i, 32):
                    self.pos_feat[0:d_model, k, i, j] = get_positional_encoding(d_model, k+1)
                    self.pos_feat[d_model:(d_model*2), k, i, j] = get_positional_encoding(d_model, i+1)
                    self.pos_feat[(d_model*2):(d_model*3), k, i, j] = get_positional_encoding(d_model, j+1)
        self.pos_feat = self.pos_feat.cuda()
        self.word_pos_encoder = SinusoidalPositionalEmbedding(512, 0, 20)
        self.mask_vec = nn.Parameter(torch.zeros(512).float(), requires_grad=True)
        # self.c_l = nn.Conv1d(in_channels=16,out_channels=512,kernel_size=1)
        self.conv1x1=nn.Linear(1024,512)
        # self.gg=nn.Linear(1024,512)
        # self.trans = DualTransformer()
        self.avg_pool = nn.AvgPool1d(8, 8)
        self.DynamicGCNFusion=DynamicGCNFusion()
        #############################################################文本编码器参数设置#############################################
        BIDIRECTIONA=False
        BIDIRECTIONAL=False
        hidden_size = 512
        txt_input_size = 300
        txt_hidden_size = 512
        self.textual_encoder = nn.LSTM(txt_input_size, txt_hidden_size//2 if BIDIRECTIONA else txt_hidden_size,
                                       num_layers=3, bidirectional=BIDIRECTIONAL, batch_first=True)
        self.tex_linear = nn.Linear(txt_hidden_size, hidden_size)

  ##############################################################################################################################
        self.fc1=nn.Linear(500, 512)
        self.fc2 = nn.Linear(512, 8001)
    # textual_input, textual_mask, sentence_mask, visual_input, duration, weights_list, ids_list
    def forward(self, textual_input, textual_mask, sentence_mask, visual_input,duration,weights_list,ids_list):


        aa= textual_mask.size(2)
        batch_size = textual_input.size(0)
        textual_mask2=textual_mask.squeeze().view(batch_size*8,aa)
        vis_h,visual_output = self.frame_layer(visual_input.transpose(1, 2)) #vis_h (b,512,64)
        #####################################################################################################
        #visual_output_g b,512,32
        visual_output_g=vis_h.permute(2, 0, 1)
        # print(visual_output_g.shape)torch.Size([32, 16, 512])
        video_mask = torch.full((batch_size, 256), False).cuda()
        tgt_src_video = self.reg_token3.weight.unsqueeze(1).repeat(1, batch_size, 1)
        tgt_mask_video= torch.zeros((batch_size, 1)).to(torch.bool).cuda()
        vl_pos_video = self.vl_pos_embed3.weight.unsqueeze(1).repeat(1, batch_size, 1)
        vl_src_video = torch.cat([tgt_src_video,visual_output_g], dim=0)
        vl_mask_video = torch.cat([tgt_mask_video, video_mask], dim=1)
        vg_hs_video = self.vl_transformer1(vl_src_video, vl_mask_video, vl_pos_video)  # (1+L+N)xBxC
        vis_h = vg_hs_video[1:, :, :].permute(1, 2, 0)
        vg_hs_video = vg_hs_video[0]
    #####################################################################################################################
        vis_h = self.avg_pool(vis_h)
####################################################################################################################
        map_h, _ = self.prop_layer(vis_h) #map_h (b,512,64,64) map_mask (b,1,64,64)
        # map_h = self.bmn_layer(vis_h)
        map_size = map_h.size(3)
        map_mask_1=np.zeros((batch_size,1,32,32))
        for i in range(batch_size):
            np.fill_diagonal(map_mask_1[i,0],1)
            map_mask_1[i,0] = np.triu(np.ones((32,32)))
        map_mask=torch.from_numpy(map_mask_1).cuda().float()
#####################################################################################将文本编码器移除到外面###################################################

        self.textual_encoder.flatten_parameters()
        batch_size = textual_input.size(0)
        seq = textual_input.size(2)

        # To LSTM
        # Single sentence:
        # txt_h = self.textual_encoder(textual_input)[0] * textual_mask # txt_h:(b,seq,512)
        textual_input = textual_input.view((batch_size * 8, seq, 300))  # textual_input: (b,8,seq,300)->(b*8,seq,300)
        txt_h = self.textual_encoder(textual_input)[0]  # txt_h:(b*8,seq,512)
        txt_h = txt_h.view((batch_size, 8, seq, 512))  # txt_h:(b,8,seq,512)
        txt_h = txt_h * textual_mask  # txt_h:(b,8,seq,512), textual_mask:(b,8,seq,1)
        txt_h_a=txt_h
        txt_h = txt_h.view((batch_size * 8, seq, 512))
        textual_mask1=textual_mask
        textual_mask = textual_mask.view((batch_size * 8, seq, 1))

        # get LSTM's last output
        # Single sentence:
        # txt_h = torch.stack([txt_h[i][torch.sum(mask).long() - 1] for i, mask in enumerate(textual_mask)]) # txt_h:(b,512)
        txt_h_ = torch.zeros(batch_size * 8, 512).cuda()  # txt_h_ (b*8,512)
        for i, mask in enumerate(textual_mask):
            cur_seq = torch.sum(mask).long()
            if cur_seq > 0:
                txt_h_[i] = txt_h[i][cur_seq - 1]

        # Single sentence:
        # txt_h = self.tex_linear(txt_h)[:,:,None,None] # txt_h:(b,512,1,1)
        txt_h = self.tex_linear(txt_h_)
        txt_h = txt_h.view(batch_size, 8, 512)
        txt_h = txt_h[:, :, :, None, None]#16,8,512,1,1
############################################################transformer文本特征聚合#########################################################
        #计算全局文本信息
        txt_g = txt_h.squeeze(-1).squeeze(-1).permute(1, 0, 2)

        mask_t_g=sentence_mask.squeeze(-1)
        mask_t_g = mask_t_g == 0
        # mask_t_g=mask_squeezed.cuda()
        #mask_t_g   16，8 bool
        tgt_src_t = self.reg_token2.weight.unsqueeze(1).repeat(1, batch_size, 1)
        tgt_mask_t = torch.zeros((batch_size, 1)).to(torch.bool).cuda()
        vl_pos_t = self.vl_pos_embed2.weight.unsqueeze(1).repeat(1, batch_size, 1)
        vl_src_t = torch.cat([tgt_src_t,txt_g], dim=0)
        vl_mask_t = torch.cat([tgt_mask_t, mask_t_g], dim=1)
        vg_hs_t = self.vl_transformer1(vl_src_t, vl_mask_t, vl_pos_t)  # (1+L+N)xBxC
        last_eight = vg_hs_t[-8:]
        txt_h = last_eight.transpose(0, 1).reshape(batch_size, 8, 512, 1, 1)
        vg_hs_t = vg_hs_t[0]
##############################################################################################################################################################
        fused_h, map_mask,_ = self.fusion_layer(txt_h, map_h, map_mask)
 #################################################################################通过图卷积构建不同层级查询文本###############################################
        txt_h_b=txt_h.view(batch_size , 8,512)
        batch_output_list = []  # 用于存储每次计算后的 x
        batch_output_g_list=[]
        for i in range(batch_size):
                    txt_h_b_i=txt_h_b[i,:,:]#维度8，512
                    sentence_mask_i=sentence_mask[i,:,:]
                    # print(sentence_mask_i.shape)torch.Size([8, 1])
                    # 统计有多少个1
                    a = torch.sum(sentence_mask_i).item()
                    # 生成 ids 列表
                    # 将 a 转换为整数
                    ids = [str(i) for i in range(0, int(a))]

                    # 将掩码展开为一维并应用掩码，取出掩码为1的特征
                    masked_features = txt_h_b_i[sentence_mask_i.squeeze(1) == 1]

                    result = None
                    while result is None:
                        result = generate_group_layers(ids)

                    group_layers_list = convert_layers_to_list(result)
                    group_layers_list = [[group[0]] for group in group_layers_list]
                    # print(result)
                    # x = masked_features[0,:].unsqueeze(0).cuda()
                    x = masked_features.cuda()
                    # print(x.shape)每个段落中的句子的文本特征
                    output_list = []  # 用于存储每次计算后的 x
                    for index, layer_groups in enumerate(group_layers_list):
                        if index == 0:
                            x=x
                        edge_index = create_edge_index(layer_groups)
                        edge_index=edge_index.cuda()
                        # print(x.shape)torch.Size([5, 512])
                        x = self.DynamicGCNFusion(x, edge_index)
                        # print(x.shape)
                        output_list.append(x)  # 将计算的 x 添加到列表中，使用 clone() 防止原值改变
                    # 将 output_list 转换为一个 tensor
                    output_tensor = torch.stack(output_list)  # 转换为 tensor
                    # print(output_tensor.shape)
                    last_vector = output_tensor[0, -1, :]
                    batch_output_g_list.append(last_vector)

                    # 获取当前张量的形状
                    current_shape = output_tensor.shape
                    # 需要填充的目标大小为 (8, 8, 512)
                    target_size = (8, 8, current_shape[2])
                    # 计算需要填充的大小
                    padding_height = target_size[0] - current_shape[0]  # 第一个维度的填充量
                    padding_width = target_size[1] - current_shape[1]  # 第二个维度的填充量
                    # 如果填充量大于0才进行填充
                    if padding_height > 0 or padding_width > 0:
                        padded_tensor = F.pad(output_tensor, (0, 0, 0, padding_width, 0, padding_height), "constant", 0)
                    else:
                        padded_tensor = output_tensor  # 如果已经是 8x8 的张量，直接返回
                    # print(padded_tensor.shape)  # 输出 (8, 8, 512)
                    batch_output_list.append(padded_tensor)
        batch_output_tensor = torch.stack( batch_output_list)  # 转换为 tensor
        # print(batch_output_tensor.shape)
        vg_hs_t_layers = torch.stack(batch_output_g_list, dim=0)
        # print(batch_output_tensor.shape)torch.Size([16, 8, 8, 512])
        #########################################################################################################################################################
        # fused_h (b, K,512,64,64)
        # print(txt_h_a.shape)torch.Size([2, 8, 24, 512])
        fused_h = fused_h.reshape(batch_size * 8, 512, map_size, map_size)  # fused_h (b*8,512,64,64)
        map_mask = map_mask.view(batch_size * 8, 1, map_size, map_size)
        sentence_mask = sentence_mask.view(batch_size * 8, 1)
        sentence_mask = sentence_mask[:, :, None, None]  # sentence_mask (b*8,1, 1, 1)
        map_mask = map_mask * sentence_mask
        # different due to conv3d
        # map_mask (b*8,1,64,64) -> (b,1,8,64,64)
        # fused_h  (b*8,512,64,64) -> (b,512,8,64,64)
        map_mask = map_mask.view(batch_size, 8, 1, map_size, map_size)
        map_mask = map_mask.permute(0, 2, 1, 3, 4)
        fused_h = fused_h.view(batch_size, 8, 512, map_size, map_size)
        fused_h = fused_h.permute(0, 2, 1, 3, 4)
        fused_h = torch.cat((self.pos_feat.repeat(fused_h.size(0), 1, 1, 1, 1).cuda(), fused_h), dim=1)
        # embed()
        fused_h = self.map_layer(fused_h, map_mask) #fused_h (b,512,8,64,64)
        # fused_h = self.nlblock(fused_h) * map_mask
        # different due to conv3d
        # map_mask (b,1,8,64,64)  -> (b*8,1,64,64)
        # fused_h  (b,512,8,64,64) -> (b*8,512,64,64)
        fused_h = fused_h.permute(0, 2, 1, 3, 4)
        map_mask = map_mask.permute(0, 2, 1, 3, 4)

        fused_h = fused_h.contiguous().view(batch_size*8, 512, map_size, map_size)
        map_mask = map_mask.contiguous().view(batch_size*8, 1, map_size, map_size)
        # print(fused_h.shape)torch.Size([2，8, 512, 32, 32])
        prediction = self.pred_layer1(fused_h)  #prediction (b*K,1,64,64)
        fused_h_t=fused_h.clone()
        prediction = prediction * map_mask #prediction (b*K,1,64,64)
        prediction = prediction.view(batch_size, 8, 1, map_size, map_size)
        # print(prediction.shape)torch.Size([2, 8, 1, 32, 32])
        map_mask = map_mask.view(batch_size, 8, 1, map_size, map_size)
        # print(map_mask.shape)torch.Size([2, 8, 1, 32, 32])上三角为1，下三角全是0。
        tmp_shape = prediction.shape
################################################################################计算最佳的提案所对应的融合特征#####################################################################
        joint_prob = torch.sigmoid(prediction) * map_mask
        weight_1, targets_tmp = torch.max(joint_prob.flatten(-2), dim=-1)
        values, indices = torch.topk(joint_prob.flatten(-2), 3, dim=-1)
        mask3=sentence_mask.view(batch_size,8).bool().unsqueeze(-1).unsqueeze(-1).expand_as(indices)
        indices= (indices*mask3).squeeze(2)
        targets = torch.zeros(tmp_shape[0], tmp_shape[1], tmp_shape[-2] * tmp_shape[-1]).cuda()
        targets.scatter_(2, targets_tmp, 1)
        targets = torch.reshape(targets, tmp_shape) * map_mask
        non_zero_indices = torch.nonzero(targets)
        fused_h_reshaped = fused_h.view(batch_size, 8, 512, 32, 32)
        results = torch.zeros(batch_size, 8, 512).cuda()
        mask = torch.ones(batch_size, 8, 1, dtype=torch.bool).cuda()
        for index in non_zero_indices:
            # 提取特定索引的元素
            element = fused_h_reshaped[index[0], index[1], :, index[3], index[4]]
            # 填充到结果容器中
            results[index[0], index[1]] = element
            # 更新掩码：已填充的位置设置为 False
            mask[index[0], index[1], 0] = False
        # print(results.shape)torch.Size([2, 8, 512])
        mask_squeezed = torch.squeeze(mask, dim=2)
        results = results.view(8,batch_size,512)
############################################################################################################################
        tgt_src_v = self.reg_token1.weight.unsqueeze(1).repeat(1, batch_size, 1)
        tgt_mask_v = torch.zeros((batch_size, 1)).to(tgt_src_v.cuda()).to(torch.bool)
        vl_pos_v = self.vl_pos_embed1.weight.unsqueeze(1).repeat(1, batch_size, 1)
        vl_src = torch.cat([tgt_src_v,results.cuda()], dim=0)
        vl_mask = torch.cat([tgt_mask_v, mask_squeezed.cuda()], dim=1)
        vg_hs_v = self.vl_transformer1(vl_src, vl_mask, vl_pos_v)  # (1+L+N)xBxC
        vg_hs_v = vg_hs_v[0]
#######################################################################################################################################################################

##############################################################################################################

###################################################################################################
        vg_hs_v=torch.cat((vg_hs_video,vg_hs_v) ,dim=1)
        vg_hs_v=self.conv1x1(vg_hs_v)
        vg_hs_v = vg_hs_v / vg_hs_v.norm(dim=-1, keepdim=True)
        vg_hs_t = vg_hs_t / vg_hs_t.norm(dim=-1, keepdim=True)
        vg_hs_video = vg_hs_video / vg_hs_video.norm(dim=-1, keepdim=True)
        sims= torch.matmul(vg_hs_v,vg_hs_t.T)
        sims2 = torch.matmul(vg_hs_video, vg_hs_t.T)
        sims3 = torch.matmul(vg_hs_video, vg_hs_v.T)
 ####################################################################################################################################################
        # jj_3,weight_3=get_max_score_timestamps(joint_prob, duration)


        jj_9,weight_9  = get_max_score_timestamps_9(joint_prob, duration)

        num = jj_9.shape[2]

        jj_3=jj_9
        weight_3=weight_9
        jj_3=jj_3.view(batch_size * 8,num,2)
        weight_3=weight_3.view(batch_size * 8,num).squeeze(1)
        duration_tensor = torch.tensor(duration).cuda()
        duration_tensors = duration_tensor.repeat_interleave(8) * sentence_mask.view(batch_size * 8)
        props_feat, props_len, props_mask = _generate_proposals_feat(visual_input.unsqueeze(1).repeat(1,8,1,1).view(batch_size * 8,256,500), jj_3,duration_tensors)
#############################################################掩码文本特征#################################################
        words_pos = self.word_pos_encoder(txt_h_a.view(batch_size * 8,aa,512))
        words_len = textual_mask1.sum(dim=2, keepdim=True).view(batch_size * 8)
        # 对每个批次中的每个文本权重进行归一化处理
        for batch_index in range(weights_list.shape[0]):  # 遍历批次
            for text_index in range(weights_list.shape[1]):  # 遍历文本
                # 计算当前文本的权重总和
                weight_sum = weights_list[batch_index, text_index, :, :].sum()
                # 如果权重总和不为零，则进行归一化
                if weight_sum > 0:
                    weights_list[batch_index, text_index, :, :] /= weight_sum
        words_feat = self._mask_words(txt_h_a.view(batch_size * 8,aa,512), words_len,aa, weights=weights_list.view(batch_size * 8,aa)) + words_pos
        words_feat1 = words_feat
        ids_list = ids_list.view(batch_size * 8,aa)
        num_proposals = num
        bsz = batch_size * 8
        words_mask1 = textual_mask2.unsqueeze(1) \
            .expand(bsz, num_proposals, -1).contiguous().view(bsz * num_proposals, -1)
        # print(words_mask1.shape)torch.Size([64, 24])
        ids_list = ids_list.unsqueeze(1) \
            .expand(bsz, num_proposals, -1).contiguous().view(bsz * num_proposals, -1)
        words_feat1 = words_feat1.unsqueeze(1) \
            .expand(bsz, num_proposals, -1, -1).contiguous().view(bsz * num_proposals, words_mask1.size(1), -1)
        props_feat1=props_feat
        props_feat1=self.fc1(props_feat1)
        # fc_comp = nn.Linear(512, 8001).cuda()
        _, h = trans(props_feat1, props_mask, words_feat1, words_mask1, decoding=2)
        words_logit = self.fc2(h)
        weights=None

        return prediction, map_mask, sims, self.logit_scale,sims2, self.logit_scale2,sims3,self.logit_scale3, jj_3, weight_3, words_logit, ids_list, weights, words_mask1, vg_hs_video,vg_hs_t,vg_hs_v,batch_output_tensor

    def _mask_words(self, words_feat, words_len, aa,weights=None):
        token = self.mask_vec.cuda().unsqueeze(0).unsqueeze(0)

        masked_words = []
        for i, l in enumerate(words_len):
            if l > 0:  # 跳过长度为0的文本
                l = int(l)
                num_masked_words = l // 3  # 计算要掩码的单词数量
                masked_word = torch.zeros([aa], dtype=torch.uint8).cuda()  # 初始化掩码
                p = weights[i, :l].cpu().numpy()  # 使用weights的实际长度l
                choices = np.random.choice(np.arange(1, l + 1), num_masked_words, replace=False,
                                           p=p / p.sum())  # 根据weights随机选择

                masked_word[choices - 1] = 1  # 更新掩码，注意索引调整
                masked_words.append(masked_word)
            else:  # 对于长度为0的文本，添加全零掩码
                masked_words.append(torch.zeros([aa], dtype=torch.uint8).cuda())
        # 转换masked_words为张量并调整维度以匹配words_feat
        masked_words = torch.stack(masked_words, 0).unsqueeze(-1)
        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
        masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
        words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec
        return words_feat1




class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        import math
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, **kwargs):
        bsz, seq_len, _ = input.size()
        max_pos = seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.cuda(input.device)[:max_pos]
        return self.weights.unsqueeze(0)

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number




