# import torch
# from torch import nn
# import torch.nn.functional as F
# from IPython import embed
# import random
# import time
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
#
# class BaseFusion(nn.Module):
#
#     def __init__(self, cfg):
#         super(BaseFusion, self).__init__()
#         self.cfg = cfg
#         hidden_size = cfg.HIDDEN_SIZE
#         txt_input_size = cfg.TXT_INPUT_SIZE
#         txt_hidden_size = cfg.TXT_HIDDEN_SIZE
#         self.textual_encoder = nn.LSTM(txt_input_size, txt_hidden_size//2 if cfg.LSTM.BIDIRECTIONAL else txt_hidden_size,
#                                        num_layers=cfg.LSTM.NUM_LAYERS, bidirectional=cfg.LSTM.BIDIRECTIONAL, batch_first=True)
#         self.tex_linear = nn.Linear(txt_hidden_size, hidden_size)
#         self.vis_conv = nn.Conv2d(hidden_size, hidden_size, 1, 1)
#
#     def forward(self, textual_input, textual_mask, sentence_mask, map_h, map_mask):
#
#         # print("textual_input: ", textual_input.shape)textual_input:  torch.Size([16, 8, 44, 300])
#         # textual_input:(b,8,seq,300）其中seq是变化的，取到了最大值
#
#         # textual_mask: (b,seq,1)这个是每个句子的掩码textual_mask:  torch.Size([16, 8, 44, 1])
#         # print("textual_mask: ", textual_mask.shape)
#         # map_h: (b, 512, 64, 64)
#         # map_mask: (b,1,64,64)
#         self.textual_encoder.flatten_parameters()
#         batch_size = textual_input.size(0)
#         seq = textual_input.size(2)
#         print(seq)
# #################################################################################构建不同layer的文本查询#########################################
#         for i in range(batch_size):
#
#             sentence_mask_i = sentence_mask[i, :, :]
#
#             a = torch.sum(sentence_mask_i).item()
#             # 生成 ids 列表
#             # 将 a 转换为整数
#             ids = [str(i) for i in range(0, int(a))]
#
#             result = None
#             while result is None:
#                 result = generate_group_layers(ids)
#
#             group_layers_list = convert_layers_to_list(result)
#             textual_input_p = textual_input[i, :, :,:]
#             textual_mask_p = textual_mask[i, :,:]
#             # print(textual_mask_p.shape)(8,seq,1)
#             # print(textual_input_p.shape)(8,seq,300)
#             textual_input_p_1 =  textual_input_p[:int(a), :, :]
#             textual_mask_p_1 = textual_mask_p[:int(a), :, :]
#             for index, layer_groups in enumerate(group_layers_list):
#                 if index == 0:
#                     continue
#                 # print(textual_input_p_1.shape)
#                 # 存储所有需要拼接的结果
#                 layer_tensors_list = []
#                 for j in range(len(layer_groups)):
#
#                     layer_groups_1=layer_groups[j]
#                     # 选择对应的句子
#                     selected_embeddings = textual_input_p_1[layer_groups_1]
#                     selected_masks = textual_mask_p_1[layer_groups_1]
#                     # 应用掩码，掩码需要扩展到embedding的最后一个维度
#
#                     masked_embeddings = selected_embeddings * selected_masks.expand(-1, -1, 300)
#                     # 收集所有通过掩码为1的位置的embeddings
#                     filtered_embeddings = []
#                     for i in range(len(layer_groups_1)):
#                        # 使用掩码提取掩码值为1的embedding部分
#                        # print(masked_embeddings[i].shape)
#                        valid_embeddings = masked_embeddings[i][selected_masks[i].squeeze(-1) == 1]
#                        # print(valid_embeddings.shape)
#                        filtered_embeddings.append(valid_embeddings)
#
#                 # 在seq维度上拼接这些filtered embeddings
#                 concatenated_embeddings = torch.cat(filtered_embeddings, dim=0)#单层单个拼接后的embedding
#
#                 # 打印拼接后的结果的形状
#                 # print(concatenated_embeddings.shape)
#
#
#
#                 print('#######################')
#                 # print(sequence_input.shape)
#                 # print(sequence_input.shape)
# ###############################################################################################################################################
#         # To LSTM
#         # Single sentence:
#         # txt_h = self.textual_encoder(textual_input)[0] * textual_mask # txt_h:(b,seq,512)
#         textual_input = textual_input.view((batch_size * 8, seq, 300))  # textual_input: (b,8,seq,300)->(b*8,seq,300)
#         txt_h = self.textual_encoder(textual_input)[0]  # txt_h:(b*8,seq,512)
#         # print(txt_h.shape)
#         txt_h = txt_h.view((batch_size, 8, seq, 512))  # txt_h:(b,8,seq,512)
#         txt_h = txt_h * textual_mask  # txt_h:(b,8,seq,512), textual_mask:(b,8,seq,1)
#         txt_h_a=txt_h
#         txt_h = txt_h.view((batch_size * 8, seq, 512))
#         textual_mask = textual_mask.view((batch_size * 8, seq, 1))
#
#         # get LSTM's last output
#         # Single sentence:
#         # txt_h = torch.stack([txt_h[i][torch.sum(mask).long() - 1] for i, mask in enumerate(textual_mask)]) # txt_h:(b,512)
#         txt_h_ = torch.zeros(batch_size * 8, 512).cuda()  # txt_h_ (b*8,512)
#         for i, mask in enumerate(textual_mask):
#             cur_seq = torch.sum(mask).long()
#             if cur_seq > 0:
#                 txt_h_[i] = txt_h[i][cur_seq - 1]
#
#         # Single sentence:
#         # txt_h = self.tex_linear(txt_h)[:,:,None,None] # txt_h:(b,512,1,1)
#         txt_h = self.tex_linear(txt_h_)
#         txt_h = txt_h.view(batch_size, 8, 512)
#         txt_h = txt_h[:, :, :, None, None]
#         # print(txt_h.shape)torch.Size([4, 8, 512, 1, 1])
#         # fusion_layer: Vision
#         # Single sentence:
#         # map_h = self.vis_conv(map_h)  # map_h: (b, 512, 64, 64)
#
#         ##########################################################################################对视频特征处理##################################
#
#         map_h = self.vis_conv(map_h)  # map_h: (b, 512, 64, 64)
#         map_h = map_h.unsqueeze(1).repeat((1, 8, 1, 1, 1))  # map_h (b,8,512,64,64)
#         map_h_c=map_h
#         map_mask = map_mask.unsqueeze(1).repeat((1, 8, 1, 1, 1))  # map_mask (b,8,1,64,64)
#
#         # fusion_layer: Fusion
#         # Single sentence:
#         # fused_h = F.normalize(txt_h * map_h) * map_mask
#         # print(txt_h.shape)
#         # print(map_h.shape)
#         # torch.Size([4, 8, 512, 1, 1])
#         # torch.Size([4, 8, 512, 32, 32])
#         fused_h = F.normalize(txt_h * map_h, dim=2) * map_mask # fused_h (b,8,512,64,64)
#         return fused_h, map_mask,txt_h,txt_h_a,map_h_c
#

#
#
#
#
#
#
#
# import torch
# from torch import nn
# import torch.nn.functional as F
# from IPython import embed
# import random
# import time
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
#
# class BaseFusion(nn.Module):
#
#     def __init__(self, cfg):
#         super(BaseFusion, self).__init__()
#         self.cfg = cfg
#         hidden_size = cfg.HIDDEN_SIZE
#         txt_input_size = cfg.TXT_INPUT_SIZE
#         txt_hidden_size = cfg.TXT_HIDDEN_SIZE
#         self.textual_encoder = nn.LSTM(txt_input_size, txt_hidden_size//2 if cfg.LSTM.BIDIRECTIONAL else txt_hidden_size,
#                                        num_layers=cfg.LSTM.NUM_LAYERS, bidirectional=cfg.LSTM.BIDIRECTIONAL, batch_first=True)
#         self.tex_linear = nn.Linear(txt_hidden_size, hidden_size)
#         self.vis_conv = nn.Conv2d(hidden_size, hidden_size, 1, 1)
#
#     def forward(self, textual_input, textual_mask, sentence_mask, map_h, map_mask):
#
#         # print("textual_input: ", textual_input.shape)textual_input:  torch.Size([16, 8, 44, 300])
#         # textual_input:(b,8,seq,300）其中seq是变化的，取到了最大值
#
#         # textual_mask: (b,seq,1)这个是每个句子的掩码textual_mask:  torch.Size([16, 8, 44, 1])
#         # print("textual_mask: ", textual_mask.shape)
#         # map_h: (b, 512, 64, 64)
#         # map_mask: (b,1,64,64)
#         self.textual_encoder.flatten_parameters()
#         batch_size = textual_input.size(0)
#         seq = textual_input.size(2)
#         # print(seq)
# #################################################################################构建不同layer的文本查询#########################################
#         batch_list=[]
#         batch_list_last=[]
#         for i in range(batch_size):
#
#             sentence_mask_i = sentence_mask[i, :, :]
#
#             a = torch.sum(sentence_mask_i).item()
#             # 生成 ids 列表
#             # 将 a 转换为整数
#             ids = [str(i) for i in range(0, int(a))]
#
#             result = None
#             while result is None:
#                 result = generate_group_layers(ids)
#
#             group_layers_list = convert_layers_to_list(result)
#             textual_input_p = textual_input[i, :, :,:]
#             textual_mask_p = textual_mask[i, :,:]
#             # print(textual_mask_p.shape)(8,seq,1)
#             # print(textual_input_p.shape)(8,seq,300)
#             textual_input_p_1 =  textual_input_p[:int(a), :, :]
#             textual_mask_p_1 = textual_mask_p[:int(a), :, :]
#             layer_tensors_list_all=[]
#             for index, layer_groups in enumerate(group_layers_list):
#                 # if index == 0:
#                 #     continue
#                 # print(textual_input_p_1.shape)
#                 # 存储所有需要拼接的结果
#                 # print(len(layer_groups))
#                 layer_tensors_list = []
#                 for j in range(len(layer_groups)):#在其中一层layer进行循环
#
#                     layer_groups_1=layer_groups[j]
#                     # 选择对应的句子
#                     selected_embeddings = textual_input_p_1[layer_groups_1]
#                     selected_masks = textual_mask_p_1[layer_groups_1]
#                     # 应用掩码，掩码需要扩展到embedding的最后一个维度
#
#                     masked_embeddings = selected_embeddings * selected_masks.expand(-1, -1, 300)
#                     # 收集所有通过掩码为1的位置的embeddings
#                     filtered_embeddings = []
#                     for i in range(len(layer_groups_1)):
#                        # 使用掩码提取掩码值为1的embedding部分
#                        # print(masked_embeddings[i].shape)
#                        valid_embeddings = masked_embeddings[i][selected_masks[i].squeeze(-1) == 1]
#                        # print(valid_embeddings.shape)
#                        filtered_embeddings.append(valid_embeddings)
#
#                      # 在seq维度上拼接这些filtered embeddings
#                     concatenated_embeddings = torch.cat(filtered_embeddings, dim=0)#单层单个拼接后的embedding
#                     concatenated_embeddings=concatenated_embeddings.unsqueeze(0)
#                     txt_h1 = self.textual_encoder(concatenated_embeddings)[0]
#                     txt_h1 = self.tex_linear(txt_h1)
#                     txt_h1= txt_h1[:,-1,:]
#                     layer_tensors_list.append(txt_h1)
#                 batch_output_tensor = torch.stack(layer_tensors_list).squeeze(1) # 转换为 tensor
#                 layer_tensors_list_all.append(batch_output_tensor)
#             batch_output_tensor_all = torch.stack(layer_tensors_list_all)
#             # print(batch_output_tensor_all)
#             batch_output_tensor_last= batch_output_tensor_all[-1,0,:]
#             # print(batch_output_tensor_last.shape)
#             b = batch_output_tensor_all.shape[0]  # 获取当前的b值
#             # 计算在第一维和第二维需要填充的数量
#             pad_size = 8 - b
#             # 进行填充，确保填充后尺寸为 [8, 8, 512]
#             padded_tensor = F.pad(batch_output_tensor_all, (0, 0, pad_size, 0, pad_size, 0), "constant", 0)
#             batch_list.append(padded_tensor)
#             batch_list_last.append(batch_output_tensor_last)
#         batch_tensor = torch.stack(batch_list)
#         batch_tensor_last = torch.stack(batch_list_last)
#         # print(batch_tensor)
# ###############################################################################################################################################
#         # To LSTM
#         # Single sentence:
#         # txt_h = self.textual_encoder(textual_input)[0] * textual_mask # txt_h:(b,seq,512)
#         textual_input = textual_input.view((batch_size * 8, seq, 300))  # textual_input: (b,8,seq,300)->(b*8,seq,300)
#         txt_h = self.textual_encoder(textual_input)[0]  # txt_h:(b*8,seq,512)
#         # print(txt_h.shape)
#         txt_h = txt_h.view((batch_size, 8, seq, 512))  # txt_h:(b,8,seq,512)
#         txt_h = txt_h * textual_mask  # txt_h:(b,8,seq,512), textual_mask:(b,8,seq,1)
#         txt_h_a=txt_h
#         txt_h = txt_h.view((batch_size * 8, seq, 512))
#         textual_mask = textual_mask.view((batch_size * 8, seq, 1))
#
#         # get LSTM's last output
#         # Single sentence:
#         # txt_h = torch.stack([txt_h[i][torch.sum(mask).long() - 1] for i, mask in enumerate(textual_mask)]) # txt_h:(b,512)
#         txt_h_ = torch.zeros(batch_size * 8, 512).cuda()  # txt_h_ (b*8,512)
#         for i, mask in enumerate(textual_mask):
#             cur_seq = torch.sum(mask).long()
#             if cur_seq > 0:
#                 txt_h_[i] = txt_h[i][cur_seq - 1]
#
#         # Single sentence:
#         # txt_h = self.tex_linear(txt_h)[:,:,None,None] # txt_h:(b,512,1,1)
#         txt_h = self.tex_linear(txt_h_)
#         txt_h = txt_h.view(batch_size, 8, 512)
#         txt_h = txt_h[:, :, :, None, None]
#         # print(txt_h.shape)torch.Size([4, 8, 512, 1, 1])
#         # fusion_layer: Vision
#         # Single sentence:
#         # map_h = self.vis_conv(map_h)  # map_h: (b, 512, 64, 64)
#
#         ##########################################################################################对视频特征处理##################################
#
#         map_h = self.vis_conv(map_h)  # map_h: (b, 512, 64, 64)
#         map_h = map_h.unsqueeze(1).repeat((1, 8, 1, 1, 1))  # map_h (b,8,512,64,64)
#         map_h_c=map_h
#         map_mask = map_mask.unsqueeze(1).repeat((1, 8, 1, 1, 1))  # map_mask (b,8,1,64,64)
#
#         # fusion_layer: Fusion
#         # Single sentence:
#         # fused_h = F.normalize(txt_h * map_h) * map_mask
#         # print(txt_h.shape)
#         # print(map_h.shape)
#         # torch.Size([4, 8, 512, 1, 1])
#         # torch.Size([4, 8, 512, 32, 32])
#         fused_h = F.normalize(txt_h * map_h, dim=2) * map_mask # fused_h (b,8,512,64,64)
#         return fused_h, map_mask,txt_h,txt_h_a,map_h_c,batch_tensor, batch_tensor_last
#
import torch
from torch import nn
import torch.nn.functional as F
from IPython import embed

class BaseFusion(nn.Module):

    def __init__(self, cfg):
        super(BaseFusion, self).__init__()
        self.cfg = cfg
        hidden_size = cfg.HIDDEN_SIZE
        # txt_input_size = cfg.TXT_INPUT_SIZE
        # txt_hidden_size = cfg.TXT_HIDDEN_SIZE
        # self.textual_encoder = nn.LSTM(txt_input_size, txt_hidden_size//2 if cfg.LSTM.BIDIRECTIONAL else txt_hidden_size,
        #                                num_layers=cfg.LSTM.NUM_LAYERS, bidirectional=cfg.LSTM.BIDIRECTIONAL, batch_first=True)
        # self.tex_linear = nn.Linear(txt_hidden_size, hidden_size)
        self.vis_conv = nn.Conv2d(hidden_size, hidden_size, 1, 1)

    def forward(self, txt_h, map_h, map_mask):
        # # textual_input:(b,8,seq,300)
        # # textual_mask: (b,seq,1)
        # # map_h: (b, 512, 64, 64)
        # # map_mask: (b,1,64,64)
        # self.textual_encoder.flatten_parameters()
        # batch_size = textual_input.size(0)
        # seq = textual_input.size(2)
        #
        # # To LSTM
        # # Single sentence:
        # # txt_h = self.textual_encoder(textual_input)[0] * textual_mask # txt_h:(b,seq,512)
        # textual_input = textual_input.view((batch_size * 8, seq, 300))  # textual_input: (b,8,seq,300)->(b*8,seq,300)
        # txt_h = self.textual_encoder(textual_input)[0]  # txt_h:(b*8,seq,512)
        # txt_h = txt_h.view((batch_size, 8, seq, 512))  # txt_h:(b,8,seq,512)
        # txt_h = txt_h * textual_mask  # txt_h:(b,8,seq,512), textual_mask:(b,8,seq,1)
        # txt_h_a=txt_h
        # txt_h = txt_h.view((batch_size * 8, seq, 512))
        # textual_mask = textual_mask.view((batch_size * 8, seq, 1))
        #
        # # get LSTM's last output
        # # Single sentence:
        # # txt_h = torch.stack([txt_h[i][torch.sum(mask).long() - 1] for i, mask in enumerate(textual_mask)]) # txt_h:(b,512)
        # txt_h_ = torch.zeros(batch_size * 8, 512).cuda()  # txt_h_ (b*8,512)
        # for i, mask in enumerate(textual_mask):
        #     cur_seq = torch.sum(mask).long()
        #     if cur_seq > 0:
        #         txt_h_[i] = txt_h[i][cur_seq - 1]
        #
        # # Single sentence:
        # # txt_h = self.tex_linear(txt_h)[:,:,None,None] # txt_h:(b,512,1,1)
        # txt_h = self.tex_linear(txt_h_)
        # txt_h = txt_h.view(batch_size, 8, 512)
        # txt_h = txt_h[:, :, :, None, None]
        # print(txt_h.shape)torch.Size([4, 8, 512, 1, 1])
        # fusion_layer: Vision
        # Single sentence:
        # map_h = self.vis_conv(map_h)  # map_h: (b, 512, 64, 64)
        map_h = self.vis_conv(map_h)  # map_h: (b, 512, 64, 64)
        map_h = map_h.unsqueeze(1).repeat((1, 8, 1, 1, 1))  # map_h (b,8,512,64,64)
        map_h_c=map_h
        map_mask = map_mask.unsqueeze(1).repeat((1, 8, 1, 1, 1))  # map_mask (b,8,1,64,64)

        # fusion_layer: Fusion
        # Single sentence:
        # fused_h = F.normalize(txt_h * map_h) * map_mask
        # print(txt_h.shape)
        # print(map_h.shape)
        # torch.Size([4, 8, 512, 1, 1])
        # torch.Size([4, 8, 512, 32, 32])
        fused_h = F.normalize(txt_h * map_h, dim=2) * map_mask # fused_h (b,8,512,64,64)
        return fused_h, map_mask,map_h_c

