from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import sigmoid

import _init_paths
import os
import pprint
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, '/home/l/data_2/wmz/1_c/DepNet_ANet_Release')
from  lib.models.loss_w import weakly_supervised_loss
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
from lib.core.eval import  eval_predictions, display_results
from lib import datasets
from lib import models
from lib.core.config import config, update_config
from lib.core.engine import Engine
from lib.core.utils import AverageMeter
from lib.core import eval
from lib.core.utils import create_logger
import lib.models.loss as loss
import math
import pickle
from IPython import embed
from  lib.models.loss import bce_rescale_loss
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.autograd.set_detect_anomaly(True)
import pandas as pd

def save_to_file(data, filename):
    torch.save(data, filename)

def load_from_file(filename):
    return torch.load(filename)


def calculate_sentence_ranks_grounding(similarity_matrix, num_sentences_per_paragraph, iou_masks):
    # 段落级别的排名计算
    paragraph_ranks = []
    num_paragraphs = similarity_matrix.shape[0]

    for i in range(num_paragraphs):
        sims = similarity_matrix[i]
        sorted_indices = torch.argsort(sims, descending=True)
        target_index = sorted_indices == i
        rank = torch.where(target_index)[0] + 1
        num_sentences = int(num_sentences_per_paragraph[i].item())  # 将其转换为整数
        paragraph_ranks.extend([rank.item()] * num_sentences)  # 确保乘法操作正确


    # 生成总句子级排名列表
    ranks = torch.tensor(paragraph_ranks)

    # 创建DataFrame来保存计算结果
    results = {}

    # 定义IOU阈值列表
    iou_thresholds = [0.3, 0.5, 0.7]
    rank_levels = [1, 5, 10, 100]

    # 总句子数
    total_sentences = int(num_sentences_per_paragraph.sum().item())

    for rank_level in rank_levels:
        results[f"Rank@{rank_level}"] = {}
        for idx, iou in enumerate(iou_thresholds):
            iou_mask = iou_masks[idx, 0, :total_sentences]
            valid_ranks = ranks[iou_mask]
            results[f"Rank@{rank_level}"][f"IOU={iou}"] = (torch.sum(
                valid_ranks <= rank_level).item() / ranks.numel()) * 100

    return results





def calculate_sentence_ranks(similarity_matrix, num_sentences_per_paragraph):
    # 段落级别的排名计算
    paragraph_ranks = []
    num_paragraphs = similarity_matrix.shape[0]
    tt=num_sentences_per_paragraph.sum()
    # print('oooooo')
    # print(tt)
    # print('oooooo')
    for i in range(num_paragraphs):
        # 获取第i个段落与所有视频的相似度
        sims = similarity_matrix[i]
        # 对相似度进行降序排列并获取索引
        sorted_indices = torch.argsort(sims, descending=True)
        # 找到真正匹配的视频索引
        target_index = sorted_indices == i
        # 获取正确匹配视频的排名
        rank = torch.where(target_index)[0] + 1
        # 段落中的每个句子都被分配相同的排名
        # paragraph_ranks.extend([rank.item()] * num_sentences_per_paragraph[i].item())
        num_sentences = int(num_sentences_per_paragraph[i].item())  # 将其转换为整数
        paragraph_ranks.extend([rank.item()] * num_sentences)  # 确保乘法操作正确


    # 计算句子级Rank-1, Rank-5, Rank-10, Rank-100
    ranks = torch.tensor(paragraph_ranks)
    total_sentences = num_sentences_per_paragraph.sum().item()  # 计算所有句子的总数
    rank1 = torch.sum(ranks <= 1).item() / total_sentences
    rank5 = torch.sum(ranks <= 5).item() / total_sentences
    rank10 = torch.sum(ranks <= 10).item() / total_sentences
    rank100 = torch.sum(ranks <= 100).item() / total_sentences
    return rank1, rank5, rank10, rank100

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', default='../experiments/dense_activitynet/acnet.yaml',required=False, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--gpus', help='gpus',default='2', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--dataDir', help='data path', type=str)
    parser.add_argument('--modelDir', help='model path', type=str)
    parser.add_argument('--logDir', help='log path', type=str)
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    parser.add_argument('--tag', help='tags shown in log', type=str)
    args = parser.parse_args()

    return args

def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.dataDir:
        config.DATA_DIR = args.dataDir
    if args.modelDir:
        config.MODEL_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.verbose:
        config.VERBOSE = args.verbose
    if args.tag:
        config.TAG = args.tag





if __name__ == '__main__':

    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir = create_logger(config, args.cfg, config.TAG)
    logger.info('\n'+pprint.pformat(args))
    logger.info('\n'+pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    dataset_name = config.DATASET.NAME
    model_name = config.MODEL.NAME

    test_dataset = getattr(datasets, dataset_name)('test')

    model = getattr(models, model_name)()

    model_checkpoint = torch.load(config.MODEL.CHECKPOINT)
    model.load_state_dict(model_checkpoint,strict=True)


    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(test_dataset,
                            batch_size=config.TEST.BATCH_SIZE,
                            shuffle=False,
                            num_workers=config.WORKERS,
                            pin_memory=False,
                            collate_fn=datasets.dense_collate_fn)

    def network(sample):
        # identical as single
        # anno_idxs:(b,) list
        # visual_input: (b,256,500) tensor

        # different due to dense
        # textual_input: (b,K,seq,300) tensor
        # textual_mask: (b,K,seq,1) tensor
        # sentence_mask: (b,K,1) tensor
        # map_gt: (b,K,1,64,64) tensor

        anno_idxs = sample['batch_anno_idxs']
        textual_input = sample['batch_word_vectors'].cuda()
        textual_mask = sample['batch_txt_mask'].cuda()
        sentence_mask = sample['batch_sentence_mask'].cuda()  # new
        visual_input = sample['batch_vis_input'].cuda()
        duration = sample['batch_duration']
        sentence_mask1 = sentence_mask.squeeze()
        sentence_number = sentence_mask1.sum(dim=1).unsqueeze(1)
        bsz = 64
        temp_dir = './temp'  # 保存临时文件的目录
        os.makedirs(temp_dir, exist_ok=True)
        with torch.no_grad():
          for idx in tqdm(range(len(anno_idxs)), desc="Computing query2video scores"):
            sims_list_batch = []
            joint_prob_batch = []
            sims_list1=[]
            for i in range(0, len(anno_idxs), bsz):
                visual_input_batch = visual_input[i:i + bsz]
                batch_size = min(bsz, len(anno_idxs) - i)
                vis_h, visual_output = model.frame_layer(visual_input_batch.transpose(1, 2))

                textual_input_batch = textual_input[idx].unsqueeze(0).repeat(batch_size, 1, 1, 1)
                textual_mask_batch = textual_mask[idx].unsqueeze(0).repeat(batch_size, 1, 1, 1)
                sentence_mask_batch = sentence_mask[idx].unsqueeze(0).repeat(batch_size, 1, 1)

                visual_output_g = vis_h
                map_h, map_mask = model.prop_layer(vis_h)
                map_h = model.bmn_layer(vis_h)
                map_size = map_h.size(3)
                fused_h, map_mask, txt_h, txt_h_a, _ = model.fusion_layer(textual_input_batch, textual_mask_batch,
                                                                          sentence_mask_batch, map_h, map_mask)
                fused_h = fused_h.view(batch_size * 8, 512, map_size, map_size)
                map_mask = map_mask.view(batch_size * 8, 1, map_size, map_size)
                sentence_mask_batch = sentence_mask_batch.view(batch_size * 8, 1)[:, :, None, None]
                map_mask = map_mask * sentence_mask_batch

                map_mask = map_mask.view(batch_size, 8, 1, map_size, map_size).permute(0, 2, 1, 3, 4)
                fused_h = fused_h.view(batch_size, 8, 512, map_size, map_size).permute(0, 2, 1, 3, 4)
                fused_h = torch.cat((model.pos_feat.repeat(fused_h.size(0), 1, 1, 1, 1).cuda(), fused_h), dim=1)

                fused_h = model.map_layer(fused_h, map_mask)
                fused_h = fused_h.permute(0, 2, 1, 3, 4).contiguous().view(batch_size * 8, 512, map_size, map_size)
                map_mask = map_mask.permute(0, 2, 1, 3, 4).contiguous().view(batch_size * 8, 1, map_size, map_size)

                prediction = model.pred_layer1(fused_h) * map_mask
                prediction = prediction.view(batch_size, 8, 1, map_size, map_size)
                map_mask = map_mask.view(batch_size, 8, 1, map_size, map_size)
                tmp_shape = prediction.shape

                joint_prob = torch.sigmoid(prediction) * map_mask
                weight_1, targets_tmp = torch.max(joint_prob.flatten(-2), dim=-1)

                values, indices = torch.topk(joint_prob.flatten(-2), 3, dim=-1)
                mask3 = sentence_mask_batch.view(batch_size, 8).bool().unsqueeze(-1).unsqueeze(-1).expand_as(indices)
                indices = (indices * mask3).squeeze(2)

                first_elements = indices[:, :, 0].unsqueeze(-1)
                second_elements = indices[:, :, 1].unsqueeze(-1)
                last_elements = indices[:, :, 2].unsqueeze(-1)

                targets = torch.zeros(tmp_shape[0], tmp_shape[1], tmp_shape[-2] * tmp_shape[-1]).cuda()
                targets_second = torch.zeros(tmp_shape[0], tmp_shape[1], tmp_shape[-2] * tmp_shape[-1]).cuda()
                targets_last = torch.zeros(tmp_shape[0], tmp_shape[1], tmp_shape[-2] * tmp_shape[-1]).cuda()

                targets.scatter_(2, targets_tmp, 1)
                targets_second.scatter_(2, second_elements, 1)
                targets_last.scatter_(2, last_elements, 1)

                targets = torch.reshape(targets, tmp_shape) * map_mask
                targets_second = torch.reshape(targets_second, tmp_shape) * map_mask
                targets_last = torch.reshape(targets_last, tmp_shape) * map_mask

                non_zero_indices = torch.nonzero(targets)
                non_zero_indices_second = torch.nonzero(targets_second)
                non_zero_indices_last = torch.nonzero(targets_last)

                fused_h_reshaped = fused_h.view(batch_size, 8, 512, 32, 32)
                map_h_c = map_h.unsqueeze(1).repeat(1, 8, 1, 1, 1)

                results = torch.zeros(batch_size, 8, 512).cuda()
                results_second = torch.zeros(batch_size, 8, 512).cuda()
                results_last = torch.zeros(batch_size, 8, 512).cuda()
                mask = torch.ones(batch_size, 8, 1, dtype=torch.bool).cuda()

                for index in non_zero_indices:
                    element = fused_h_reshaped[index[0], index[1], :, index[3], index[4]]
                    results[index[0], index[1]] = element
                    mask[index[0], index[1], 0] = False

                mask_squeezed = torch.squeeze(mask, dim=2)
                results = results.view(8, batch_size, 512)

                for index in non_zero_indices_second:
                    element = fused_h_reshaped[index[0], index[1], :, index[3], index[4]]
                    results_second[index[0], index[1]] = element
                results_second = results_second.view(8, batch_size, 512)

                for index in non_zero_indices_last:
                    element = fused_h_reshaped[index[0], index[1], :, index[3], index[4]]
                    results_last[index[0], index[1]] = element
                results_last = results_last.view(8, batch_size, 512)

                results = results * 0.4 + results_second * 0.3 + results_last * 0.3

                tgt_src_v = model.reg_token1.weight.unsqueeze(1).repeat(1, batch_size, 1)
                tgt_mask_v = torch.zeros((batch_size, 1)).to(tgt_src_v.device).to(torch.bool)
                vl_pos_v = model.vl_pos_embed1.weight.unsqueeze(1).repeat(1, batch_size, 1)
                vl_src = torch.cat([tgt_src_v, results.cuda()], dim=0)
                vl_mask = torch.cat([tgt_mask_v, mask_squeezed.cuda()], dim=1)
                vg_hs_v = model.vl_transformer1(vl_src, vl_mask, vl_pos_v)[0]

                txt_g = txt_h.squeeze(-1).squeeze(-1).permute(1, 0, 2)
                mask_t_g = mask_squeezed.cuda()
                tgt_src_t = model.reg_token2.weight.unsqueeze(1).repeat(1, batch_size, 1)
                tgt_mask_t = torch.zeros((batch_size, 1)).to(tgt_src_v.device).to(torch.bool).cuda()
                vl_pos_t = model.vl_pos_embed2.weight.unsqueeze(1).repeat(1, batch_size, 1)
                vl_src_t = torch.cat([tgt_src_t, txt_g], dim=0)
                vl_mask_t = torch.cat([tgt_mask_t, mask_t_g], dim=1)
                vg_hs_t = model.vl_transformer1(vl_src_t, vl_mask_t, vl_pos_t)[0]

                visual_output_g = visual_output_g.permute(2, 0, 1)
                video_mask = torch.full((batch_size, 32), False).cuda()
                tgt_src_video = model.reg_token3.weight.unsqueeze(1).repeat(1, batch_size, 1)
                tgt_mask_video = tgt_mask_t
                vl_pos_video = model.vl_pos_embed3.weight.unsqueeze(1).repeat(1, batch_size, 1)
                vl_src_video = torch.cat([tgt_src_video, visual_output_g], dim=0)
                vl_mask_video = torch.cat([tgt_mask_video, video_mask], dim=1)
                vg_hs_video = model.vl_transformer1(vl_src_video, vl_mask_video, vl_pos_video)[0]

                vg_hs_v = torch.cat((vg_hs_video, vg_hs_v), dim=1)
                vg_hs_v = model.conv1x1(vg_hs_v)
                vg_hs_v = vg_hs_v / vg_hs_v.norm(dim=-1, keepdim=True)
                vg_hs_t = vg_hs_t / vg_hs_t.norm(dim=-1, keepdim=True)
                # vg_hs_video = vg_hs_video / vg_hs_video.norm(dim=-1, keepdim=True)
                sims2 = torch.matmul(vg_hs_v, vg_hs_t.T)
                sim2_tensor = torch.diag(sims2)
                sims_list_batch.extend(sim2_tensor.tolist())
                joint_prob1 = joint_prob.detach()
                joint_prob_batch.append(joint_prob1)
            sims_list1.append(sims_list_batch)
            joint_prob_tensor0 = torch.cat(joint_prob_batch, dim=0)
            joint_prob_tensor_idx3 = joint_prob_tensor0[idx].unsqueeze(0).clone()
            temp_sims_file = os.path.join(temp_dir, f'sims_list_{idx}.pt')
            temp_joint_prob_file = os.path.join(temp_dir, f'joint_prob_{idx}.pt')

            save_to_file(sims_list1, temp_sims_file)
            save_to_file(joint_prob_tensor_idx3, temp_joint_prob_file)
          sims_list = []
          joint_prob_list = []
          for idx in tqdm(range(len(anno_idxs)), desc="Reloading saved results"):
              temp_sims_file = os.path.join(temp_dir, f'sims_list_{idx}.pt')
              temp_joint_prob_file = os.path.join(temp_dir, f'joint_prob_{idx}.pt')

              sims_list_batch = torch.load(temp_sims_file)
              joint_prob_tensor_idx = torch.load(temp_joint_prob_file)

              sims_list.extend(sims_list_batch)
              joint_prob_list.append(joint_prob_tensor_idx)

          joint_prob_tensor = torch.cat(joint_prob_list, dim=0)
          sims_list_tensor = torch.tensor(sims_list)
          sorted_times = get_proposal_results(joint_prob_tensor, duration)

          loss_value = 1
          return loss_value, sorted_times, sims_list_tensor, sentence_number


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



    def on_test_start(state):
        # state['loss_meter'] = AverageMeter()
        state['sorted_segments_list']=[]
        state['sorted_sims_g'] = []
        state['sentence_number_list'] = []
        if config.VERBOSE:
            if state['split'] == 'test':
                state['progress_bar'] = tqdm(total=math.ceil(len(test_dataset)/config.TEST.BATCH_SIZE))
            else:
                raise NotImplementedError

    def on_test_forward(state):
        if config.VERBOSE:
            state['progress_bar'].update(1)
        # state['loss_meter'].update(state['loss'].item(), 1)

        min_idx = min(state['sample']['batch_anno_idxs'])
        batch_indexs = [idx - min_idx for idx in state['sample']['batch_anno_idxs']]
        sorted_segments = [state['output'][i] for i in batch_indexs]
        state['sorted_segments_list'].extend(sorted_segments)

        sorted_sentence_number=state['sentence_number'].cpu().detach()
        # 假设state字典已经被初始化，并且有sorted_video_list和sorted_text_list键
        # state = {'sorted_video_list': [], 'sorted_text_list': []}
        simss=state['sims_g']
        # print(simss.shape)torch.Size([64, 64])
        state['sorted_sims_g'] = simss

        for batch_index in range(sorted_sentence_number.shape[0]):  # 遍历每个批次
            # 从每个批次中提取单个512维向量

            sorted_sentence_number_vector=sorted_sentence_number[batch_index]
            # 将这个向量追加到相应的列表中

            state['sentence_number_list'].append(sorted_sentence_number_vector)

    def on_test_end(state):
        #############################################################################定位评价指标###################################################################################
        annotations = state['iterator'].dataset.annotations
        state['Rank@N,mIoU@M'], state['miou'], grounding_mask = eval.eval_predictions(state['sorted_segments_list'],
                                                                                      annotations, verbose=False)
        print('################################################################################## 检索评价指标############################################################################')

        sorted_sentence_number_list_all = state['sentence_number_list']
        sims_l=state['sorted_sims_g']
        # print(sims_l.shape)
        sorted_sentence_number_list_tensor = torch.stack(sorted_sentence_number_list_all, dim=0)


        rank1, rank5, rank10, rank100 = calculate_sentence_ranks(sims_l, sorted_sentence_number_list_tensor)
        # 将结果以表格形式展示
        results1 = pd.DataFrame({
            "Rank-1": [rank1 * 100],
            "Rank-5": [rank5 * 100],
            "Rank-10": [rank10 * 100],
            "Rank-100": [rank100 * 100]
        })
        print('使用定位增强检索特征进行检索相似度计算')
        print(results1)
        print('################################################################################## 检索加定位评价指标###########################################################################')
        r_l = calculate_sentence_ranks_grounding(sims_l, sorted_sentence_number_list_tensor, grounding_mask)
        print('使用定位增强检索特征进行检索定位')
        for rank, data in r_l.items():
            df = pd.DataFrame(data, index=[rank])
            print(df)
            print("\n" + "=" * 40 + "\n")  # 添加分隔线以区分不同的表格


        if config.VERBOSE:
            state['progress_bar'].close()

    engine = Engine()
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    engine.test(network, dataloader,split='test')