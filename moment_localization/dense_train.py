

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pprint
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, '/home/l/data_1/wmz3/DepNet_ANet_Release2')
from  lib.models.loss_w import weakly_supervised_loss
import torch.optim as optim
from tqdm import tqdm
from lib.models.loss_t import get_frame_trip_loss,calculate_mse_loss
from lib import datasets
from lib import models
from lib.core.config import config, update_config
from lib.core.engine import Engine
from lib.core.utils import AverageMeter
from lib.core import eval
from lib.core.utils import create_logger
import lib.models.loss as loss
import math
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.autograd.set_detect_anomaly(True)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from lib.models.loss_t_h import loss_t_h
from lib.models.loss_t_h import loss_t_h
from thop import profile
from lib.models.loss_m import apply_mask_and_calculate_loss

def calculate_p_rank_accuracy(text_features, video_features, video_ids, input_video_ids):
    # 计算文本和视频之间的相似度分数
    similarity_scores = torch.matmul(text_features, video_features.T)

    # 从视频库文件名中提取核心视频ID
    core_video_ids = video_ids

    # 初始化计数器
    rank1_count = 0
    rank5_count = 0
    rank10_count = 0
    rank100_count = 0
    total_videos = len(input_video_ids)  # 总视频数

    for i, true_id in enumerate(input_video_ids):
        # 获取相似度分数，排序，找到最高分数的视频索引
        sorted_indices = torch.argsort(similarity_scores[i], descending=True)
        sorted_video_ids = [core_video_ids[idx] for idx in sorted_indices]

        # 检查正确的视频是否在排序中的前1, 5, 10, 100位
        if true_id in sorted_video_ids[:1]:
            rank1_count += 1
        if true_id in sorted_video_ids[:5]:
            rank5_count += 1
        if true_id in sorted_video_ids[:10]:
            rank10_count += 1
        if true_id in sorted_video_ids[:100]:
            rank100_count += 1
            # print(rank100_count)
    # print("Rank 100 count_p:",rank100_count)
    # print("短语查询数量:",total_videos)
    # 计算准确率
    rank1_accuracy = rank1_count / total_videos
    rank5_accuracy = rank5_count / total_videos
    rank10_accuracy = rank10_count / total_videos
    rank100_accuracy = rank100_count / total_videos

    return rank1_accuracy, rank5_accuracy, rank10_accuracy, rank100_accuracy









def calculate_sentence_rank_accuracy_with_grounding(text_features, video_features, video_ids, input_video_ids,
                                                    sentences_per_paragraph_tensor, grounding_masks):
    # Calculate text-video similarity scores
    similarity_scores = torch.matmul(text_features, video_features.T)





    # Extract core video IDs from the video library
    core_video_ids = [vid.split('.')[0] for vid in video_ids]

    # Adjust input video IDs to match video library IDs
    true_video_ids = [vid.split('.')[0] for vid in input_video_ids]

    # Initialize counters for different IOU levels and rank thresholds
    rank1_count = [0, 0, 0]  # for IOU=0.3, 0.5, 0.7
    rank5_count = [0, 0, 0]
    rank10_count = [0, 0, 0]
    rank100_count = [0, 0, 0]
    total_sentences = sentences_per_paragraph_tensor.sum().item()  # Convert tensor sum to scalar

    sentence_index = 0  # Track sentence index across paragraphs

    for i, true_id in enumerate(true_video_ids):
        # Get similarity scores and sort indices in descending order
        sorted_indices = torch.argsort(similarity_scores[i], descending=True)
        sorted_video_ids = [core_video_ids[idx] for idx in sorted_indices]

        # Get the number of sentences in the current paragraph and ensure it's an integer
        sentence_count = int(sentences_per_paragraph_tensor[i].item())

        # Determine if the paragraph's retrieval was correct
        retrieval_correct = true_id in sorted_video_ids[:100]  # Assuming Rank-100 is the largest context we consider

        for j in range(sentence_count):
            # Check each IOU level
            for k in range(3):
                if retrieval_correct and true_id in sorted_video_ids[:1] and grounding_masks[k, 0, sentence_index + j]:
                    rank1_count[k] += 1
                if retrieval_correct and true_id in sorted_video_ids[:5] and grounding_masks[k, 0, sentence_index + j]:
                    rank5_count[k] += 1
                if retrieval_correct and true_id in sorted_video_ids[:10] and grounding_masks[k, 0, sentence_index + j]:
                    rank10_count[k] += 1
                if retrieval_correct and true_id in sorted_video_ids[:100] and grounding_masks[k, 0, sentence_index + j]:
                    rank100_count[k] += 1

        sentence_index += sentence_count  # Update sentence index for next iteration

    # Calculate accuracy for each IOU level at each rank threshold, convert to percentage
    rank1_accuracy = [(count / total_sentences) * 100 for count in rank1_count]  # Multiply by 100 for percentage
    rank5_accuracy = [(count / total_sentences) * 100 for count in rank5_count]
    rank10_accuracy = [(count / total_sentences) * 100 for count in rank10_count]
    rank100_accuracy = [(count / total_sentences) * 100 for count in rank100_count]

    return rank1_accuracy, rank5_accuracy, rank10_accuracy, rank100_accuracy


def calculate_sentence_rank_accuracy_with_grounding_GCN(text_features, video_features, video_ids, input_video_ids,
                                                    sentences_per_paragraph_tensor, grounding_masks):
    text_features_reshaped = text_features.reshape(4885, -1, 512)  # 调整文本特征的维度以便进行矩阵乘法
    video_features_T = video_features.permute(1, 0)  # 转置 video_features 以便进行矩阵乘法
    similarity_scores = torch.matmul(text_features_reshaped, video_features_T)  # 计算得分

    # # 检查 similarity_scores 的维度
    # print(f'similarity_scores.shape: {similarity_scores.shape}')

    # 初始化一个 tensor 用来保存最终的分数
    final_scores = torch.zeros(4885, 4885)

    # 根据 tensorA 计算每个批次的非0层的最后一层非0句子的分数，并取平均值
    for i in range(4885):
        n_layers = sentences_per_paragraph_tensor[i].item()  # 当前批次的真实非0层数
        n_layers = int(n_layers)
        last_non_zero_layer_index = n_layers - 1  # 最后一个非0层的索引

        # 计算索引范围，确保索引为整数
        start_index = int(last_non_zero_layer_index * 8)  # 每层有8个句子
        end_index = int(start_index + 8)

        # 获取最后一个非0层的分数
        last_non_zero_layer_scores = similarity_scores[i, start_index:end_index]

        # 只选取前n_layers句子的分数（因为只有这些句子是非0的）
        valid_sentence_scores = last_non_zero_layer_scores[:n_layers, :]

        # 计算这些句子分数的平均值
        final_scores[i] = valid_sentence_scores.sum(dim=0) / n_layers

    # 输出最终得分
    similarity_scores = final_scores




    # Extract core video IDs from the video library
    core_video_ids = [vid.split('.')[0] for vid in video_ids]

    # Adjust input video IDs to match video library IDs
    true_video_ids = [vid.split('.')[0] for vid in input_video_ids]

    # Initialize counters for different IOU levels and rank thresholds
    rank1_count = [0, 0, 0]  # for IOU=0.3, 0.5, 0.7
    rank5_count = [0, 0, 0]
    rank10_count = [0, 0, 0]
    rank100_count = [0, 0, 0]
    total_sentences = sentences_per_paragraph_tensor.sum().item()  # Convert tensor sum to scalar

    sentence_index = 0  # Track sentence index across paragraphs

    for i, true_id in enumerate(true_video_ids):
        # Get similarity scores and sort indices in descending order
        sorted_indices = torch.argsort(similarity_scores[i], descending=True)
        sorted_video_ids = [core_video_ids[idx] for idx in sorted_indices]

        # Get the number of sentences in the current paragraph and ensure it's an integer
        sentence_count = int(sentences_per_paragraph_tensor[i].item())

        # Determine if the paragraph's retrieval was correct
        retrieval_correct = true_id in sorted_video_ids[:100]  # Assuming Rank-100 is the largest context we consider

        for j in range(sentence_count):
            # Check each IOU level
            for k in range(3):
                if retrieval_correct and true_id in sorted_video_ids[:1] and grounding_masks[k, 0, sentence_index + j]:
                    rank1_count[k] += 1
                if retrieval_correct and true_id in sorted_video_ids[:5] and grounding_masks[k, 0, sentence_index + j]:
                    rank5_count[k] += 1
                if retrieval_correct and true_id in sorted_video_ids[:10] and grounding_masks[k, 0, sentence_index + j]:
                    rank10_count[k] += 1
                if retrieval_correct and true_id in sorted_video_ids[:100] and grounding_masks[k, 0, sentence_index + j]:
                    rank100_count[k] += 1

        sentence_index += sentence_count  # Update sentence index for next iteration

    # Calculate accuracy for each IOU level at each rank threshold, convert to percentage
    rank1_accuracy = [(count / total_sentences) * 100 for count in rank1_count]  # Multiply by 100 for percentage
    rank5_accuracy = [(count / total_sentences) * 100 for count in rank5_count]
    rank10_accuracy = [(count / total_sentences) * 100 for count in rank10_count]
    rank100_accuracy = [(count / total_sentences) * 100 for count in rank100_count]

    return rank1_accuracy, rank5_accuracy, rank10_accuracy, rank100_accuracy











def calculate_sentence_rank_accuracy(text_features, video_features, video_ids, input_video_ids, sentences_per_paragraph_tensor):
    # 计算文本和视频之间的相似度分数
    similarity_scores = torch.matmul(text_features, video_features.T)

    # 从视频库文件名中提取核心视频ID
    core_video_ids = video_ids

    # 从输入的视频ID中移除_后面的部分以匹配视频库ID
    true_video_ids = input_video_ids

    # 初始化计数器
    rank1_count = 0
    rank5_count = 0
    rank10_count = 0
    rank100_count = 0
    total_sentences = sentences_per_paragraph_tensor.sum().item()  # 将张量总和转换为标量

    for i, true_id in enumerate(true_video_ids):
        # 获取相似度分数，排序，找到最高分数的视频索引
        sorted_indices = torch.argsort(similarity_scores[i], descending=True)
        sorted_video_ids = [core_video_ids[idx] for idx in sorted_indices]

        # 句子数量从张量中获取
        sentence_count = sentences_per_paragraph_tensor[i].item()
        # print(sentence_count)
        # 检查正确的视频是否在排序中的前1, 5, 10, 100位
        if true_id in sorted_video_ids[:1]:
            rank1_count += sentence_count
        if true_id in sorted_video_ids[:5]:
            rank5_count += sentence_count
        if true_id in sorted_video_ids[:10]:
            rank10_count += sentence_count
        if true_id in sorted_video_ids[:100]:
            rank100_count += sentence_count
            # print(rank100_count)
    # print("Rank 100 count_s:",rank100_count)
    # print("句子查询数量:",total_sentences)
    rank1_accuracy = rank1_count / total_sentences
    rank5_accuracy = rank5_count / total_sentences
    rank10_accuracy = rank10_count / total_sentences
    rank100_accuracy = rank100_count / total_sentences

    return rank1_accuracy, rank5_accuracy, rank10_accuracy, rank100_accuracy


def calculate_sentence_rank_accuracy_mean(text_features, video_features, video_ids, input_video_ids, sentences_per_paragraph_tensor):
    # 计算文本和视频之间的相似度分数       # torch.Size([4885, 8, 8, 512])文本层次所有特征       视频特征4885，512

    # 计算相似度分数
    text_features_reshaped = text_features.reshape(4885, -1, 512)  # 调整文本特征的维度以便进行矩阵乘法
    video_features_T = video_features.permute(1, 0)  # 转置 video_features 以便进行矩阵乘法
    similarity_scores = torch.matmul(text_features_reshaped, video_features_T)  # 计算得分

    # # 检查 similarity_scores 的维度
    # print(f'similarity_scores.shape: {similarity_scores.shape}')

    # 初始化一个 tensor 用来保存最终的分数
    final_scores = torch.zeros(4885, 4885)

    # 根据 tensorA 计算每个批次的非0层的最后一层非0句子的分数，并取平均值
    for i in range(4885):
        n_layers = sentences_per_paragraph_tensor[i].item()  # 当前批次的真实非0层数
        n_layers = int(n_layers)
        last_non_zero_layer_index = n_layers - 1  # 最后一个非0层的索引

        # 计算索引范围，确保索引为整数
        start_index = int(last_non_zero_layer_index * 8)  # 每层有8个句子
        end_index = int(start_index + 8)

        # 获取最后一个非0层的分数
        last_non_zero_layer_scores = similarity_scores[i, start_index:end_index]

        # 只选取前n_layers句子的分数（因为只有这些句子是非0的）
        valid_sentence_scores = last_non_zero_layer_scores[:n_layers, :]

        # 计算这些句子分数的平均值
        final_scores[i] = valid_sentence_scores.sum(dim=0) / n_layers

    # 输出最终得分
    similarity_scores = final_scores
    # 从视频库文件名中提取核心视频ID
    core_video_ids = video_ids

    # 从输入的视频ID中移除_后面的部分以匹配视频库ID
    true_video_ids = input_video_ids

    # 初始化计数器
    rank1_count = 0
    rank5_count = 0
    rank10_count = 0
    rank100_count = 0
    total_sentences = sentences_per_paragraph_tensor.sum().item()  # 将张量总和转换为标量

    for i, true_id in enumerate(true_video_ids):
        # 获取相似度分数，排序，找到最高分数的视频索引
        sorted_indices = torch.argsort(similarity_scores[i], descending=True)
        sorted_video_ids = [core_video_ids[idx] for idx in sorted_indices]

        # 句子数量从张量中获取
        sentence_count = sentences_per_paragraph_tensor[i].item()
        # print(sentence_count)
        # 检查正确的视频是否在排序中的前1, 5, 10, 100位
        if true_id in sorted_video_ids[:1]:
            rank1_count += sentence_count
        if true_id in sorted_video_ids[:5]:
            rank5_count += sentence_count
        if true_id in sorted_video_ids[:10]:
            rank10_count += sentence_count
        if true_id in sorted_video_ids[:100]:
            rank100_count += sentence_count
            # print(rank100_count)
    # print("Rank 100 count_s:",rank100_count)
    # print("句子查询数量:",total_sentences)
    rank1_accuracy = rank1_count / total_sentences
    rank5_accuracy = rank5_count / total_sentences
    rank10_accuracy = rank10_count / total_sentences
    rank100_accuracy = rank100_count / total_sentences

    return rank1_accuracy, rank5_accuracy, rank10_accuracy, rank100_accuracy

def calculate_sentence_rank_accuracy_max(text_features, video_features, video_ids, input_video_ids, sentences_per_paragraph_tensor):
    # 计算文本和视频之间的相似度分数       # torch.Size([4885, 8, 8, 512])文本层次所有特征       视频特征4885，512

    # 计算相似度分数
    text_features_reshaped = text_features.reshape(4885, -1, 512)  # 调整文本特征的维度以便进行矩阵乘法
    video_features_T = video_features.permute(1, 0)  # 转置 video_features 以便进行矩阵乘法
    similarity_scores = torch.matmul(text_features_reshaped, video_features_T)  # 计算得分

    # # 检查 similarity_scores 的维度
    # print(f'similarity_scores.shape: {similarity_scores.shape}')

    # 初始化一个 tensor 用来保存最终的分数
    final_scores = torch.zeros(4885, 4885)

    # 根据 tensorA 计算每个批次的非0层的最后一层非0句子的分数，并取平均值
    for i in range(4885):
        n_layers = int(sentences_per_paragraph_tensor[i].item())  # 当前批次的真实非0层数
        last_non_zero_layer_index = n_layers - 1  # 最后一个非0层的索引

        # 计算索引范围，确保索引为整数
        start_index = last_non_zero_layer_index * 8  # 每层有8个句子
        end_index = start_index + 8

        # 获取最后一个非0层的分数
        last_non_zero_layer_scores = similarity_scores[i, start_index:end_index]

        # 只选取前n_layers句子的分数（因为只有这些句子是非0的）
        valid_sentence_scores = last_non_zero_layer_scores[:n_layers, :]

        # 计算这些句子的最大分数
        final_scores[i], _ = valid_sentence_scores.max(dim=0)  # 取最大值，并忽略索引

    # 输出最终得分
    similarity_scores = final_scores
    # 从视频库文件名中提取核心视频ID
    core_video_ids = video_ids

    # 从输入的视频ID中移除_后面的部分以匹配视频库ID
    true_video_ids = input_video_ids

    # 初始化计数器
    rank1_count = 0
    rank5_count = 0
    rank10_count = 0
    rank100_count = 0
    total_sentences = sentences_per_paragraph_tensor.sum().item()  # 将张量总和转换为标量

    for i, true_id in enumerate(true_video_ids):
        # 获取相似度分数，排序，找到最高分数的视频索引
        sorted_indices = torch.argsort(similarity_scores[i], descending=True)
        sorted_video_ids = [core_video_ids[idx] for idx in sorted_indices]

        # 句子数量从张量中获取
        sentence_count = sentences_per_paragraph_tensor[i].item()
        # print(sentence_count)
        # 检查正确的视频是否在排序中的前1, 5, 10, 100位
        if true_id in sorted_video_ids[:1]:
            rank1_count += sentence_count
        if true_id in sorted_video_ids[:5]:
            rank5_count += sentence_count
        if true_id in sorted_video_ids[:10]:
            rank10_count += sentence_count
        if true_id in sorted_video_ids[:100]:
            rank100_count += sentence_count
            # print(rank100_count)
    # print("Rank 100 count_s:",rank100_count)
    # print("句子查询数量:",total_sentences)
    rank1_accuracy = rank1_count / total_sentences
    rank5_accuracy = rank5_count / total_sentences
    rank10_accuracy = rank10_count / total_sentences
    rank100_accuracy = rank100_count / total_sentences

    return rank1_accuracy, rank5_accuracy, rank10_accuracy, rank100_accuracy




def calculate_sentence_rank_accuracy_mean_first_layer(text_features, video_features, video_ids, input_video_ids, sentences_per_paragraph_tensor):
    # 计算文本和视频之间的相似度分数       # torch.Size([4885, 8, 8, 512])文本层次所有特征       视频特征4885，512

    # 计算相似度分数
    text_features_reshaped = text_features.reshape(4885, -1, 512)  # 调整文本特征的维度以便进行矩阵乘法
    video_features_T = video_features.permute(1, 0)  # 转置 video_features 以便进行矩阵乘法
    similarity_scores = torch.matmul(text_features_reshaped, video_features_T)  # 计算得分

    # # 检查 similarity_scores 的维度
    # print(f'similarity_scores.shape: {similarity_scores.shape}')

    # 初始化一个 tensor 用来保存最终的分数
    final_scores = torch.zeros(4885, 4885)

    # 根据 tensorA 计算每个批次的非0层的最后一层非0句子的分数，并取平均值
    for i in range(4885):
        n_layers = int(sentences_per_paragraph_tensor[i].item())  # 当前批次的第一层中的非0句子数

        # 只处理第一层的前n_layers个句子
        first_layer_scores = similarity_scores[i, 0:n_layers]

        # 计算这些句子的平均分数
        final_scores[i] = first_layer_scores.mean(dim=0)



    # 输出最终得分
    similarity_scores = final_scores
    # 从视频库文件名中提取核心视频ID
    core_video_ids = video_ids

    # 从输入的视频ID中移除_后面的部分以匹配视频库ID
    true_video_ids = input_video_ids

    # 初始化计数器
    rank1_count = 0
    rank5_count = 0
    rank10_count = 0
    rank100_count = 0
    total_sentences = sentences_per_paragraph_tensor.sum().item()  # 将张量总和转换为标量

    for i, true_id in enumerate(true_video_ids):
        # 获取相似度分数，排序，找到最高分数的视频索引
        sorted_indices = torch.argsort(similarity_scores[i], descending=True)
        sorted_video_ids = [core_video_ids[idx] for idx in sorted_indices]

        # 句子数量从张量中获取
        sentence_count = sentences_per_paragraph_tensor[i].item()
        # print(sentence_count)
        # 检查正确的视频是否在排序中的前1, 5, 10, 100位
        if true_id in sorted_video_ids[:1]:
            rank1_count += sentence_count
        if true_id in sorted_video_ids[:5]:
            rank5_count += sentence_count
        if true_id in sorted_video_ids[:10]:
            rank10_count += sentence_count
        if true_id in sorted_video_ids[:100]:
            rank100_count += sentence_count
            # print(rank100_count)
    # print("Rank 100 count_s:",rank100_count)
    # print("句子查询数量:",total_sentences)
    rank1_accuracy = rank1_count / total_sentences
    rank5_accuracy = rank5_count / total_sentences
    rank10_accuracy = rank10_count / total_sentences
    rank100_accuracy = rank100_count / total_sentences

    return rank1_accuracy, rank5_accuracy, rank10_accuracy, rank100_accuracy







def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', default='/home/ps/mnt/data/data_2/wmz/three/DepNet_ANet_Release2/experiments/dense_activitynet/acnet.yaml',required=False, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--gpus', default='0', type=str)
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
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

    train_dataset = getattr(datasets, dataset_name)('train')
    if config.TEST.EVAL_TRAIN:
        eval_train_dataset = getattr(datasets, dataset_name)('train')
    if not config.DATASET.NO_VAL:
        val_dataset = getattr(datasets, dataset_name)('val')
    test_dataset = getattr(datasets, dataset_name)('test')

    model = getattr(models, model_name)()
    if config.MODEL.CHECKPOINT and config.TRAIN.CONTINUE:
        model_checkpoint = torch.load(config.MODEL.CHECKPOINT)
        model.load_state_dict(model_checkpoint)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)

    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=config.TRAIN.LR, betas=(0.9, 0.999), weight_decay=config.TRAIN.WEIGHT_DECAY)
    # optimizer = optim.SGD(model.parameters(), lr=config.TRAIN.LR, momentum=0.9, weight_decay=config.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.TRAIN.FACTOR, patience=config.TRAIN.PATIENCE, verbose=config.VERBOSE)



    def iterator(split):
        if split == 'train':
            dataloader = DataLoader(train_dataset,
                                    batch_size=config.TRAIN.BATCH_SIZE,
                                    shuffle=config.TRAIN.SHUFFLE,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.dense_collate_fn)
        elif split == 'val':
            dataloader = DataLoader(val_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.dense_collate_fn)
        elif split == 'test':
            dataloader = DataLoader(test_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.dense_collate_fn)
        elif split == 'train_no_shuffle':
            dataloader = DataLoader(eval_train_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.dense_collate_fn)
        else:
            raise NotImplementedError

        return dataloader



    def network(sample):
        # identical as single
        # anno_idxs:(b,) list
        # visual_input: (b,256,500) tensor

        # different due to dense
        # textual_input: (b,K,seq,300) tensor
        # textual_mask: (b,K,seq,1) tensor
        # sentence_mask: (b,K,1) tensor
        # map_gt: (b,K,1,64,64) tensor
        video_idxs = sample['batch_video_idxs']
        anno_idxs = sample['batch_anno_idxs']
        textual_input = sample['batch_word_vectors'].cuda()
        textual_mask = sample['batch_txt_mask'].cuda()
        sentence_mask = sample['batch_sentence_mask'].cuda()  # new
        visual_input = sample['batch_vis_input'].cuda()
        map_gt = sample['batch_map_gt'].cuda()
        # torch.Size([4, 8, 1, 32, 32])
        duration = sample['batch_duration']
        weights_list = sample['batch_weights_list']
        ids_list = sample['batch_ids_list']
        ids_list = ids_list.squeeze().int()
        # print(type(ids_list))
        # print(weights_list.shape)torch.Size([4, 8, 24, 1])
        # print(textual_mask.shape)torch.Size([4, 8, 24, 1])

        sentence_mask1=sentence_mask.squeeze()
        sentence_number=sentence_mask1.sum(dim=1).unsqueeze(1)

        flops, params = profile(model, inputs=(
        textual_input, textual_mask, sentence_mask, visual_input, duration, weights_list, ids_list))

        # 3. 输出 FLOPS 和 参数量
        print(f"FLOPs(G): {flops / 1e9}")  # 将 FLOPS 转换为 GFLOPS (10^9 FLOPS)
        print(f"Parameters: {params / 1e6} M")  # 将参数量转换为百万参数

        prediction, map_mask, sims, logit_scale, sims2, logit_scale2, sims3, logit_scale3, jj, weight_3, words_logit, ids_list, weights, words_mask1,  vg_hs_video, vg_hs_t, vg_hs_v,batch_output_tensor = model(
            textual_input, textual_mask, sentence_mask, visual_input, duration, weights_list, ids_list)

        rewards = torch.from_numpy(np.asarray([0, 0.5, 1.0])).cuda()
        loss_NCE, loss_IVC = loss_t_h(batch_output_tensor, vg_hs_video, sentence_number, logit_scale)
        # loss_value1, loss_overlap, loss_order, joint_prob = getattr(loss, config.LOSS.NAME)(prediction, map_mask, sentence_mask, overlaps_tensor.unsqueeze(2),config.LOSS.PARAMS)
        loss_mm=apply_mask_and_calculate_loss(prediction, map_mask, sentence_mask)
        # print(loss_mm)
        joint_prob = torch.sigmoid(prediction) * map_mask
        # loss_w = weakly_supervised_loss(**output, rewards=rewards)
        loss_w = weakly_supervised_loss(weight_3, words_logit, ids_list, words_mask1, rewards, sentence_mask)
        # loss_t = weakly_supervised_loss_text(words_logit, ids_list, words_mask1)
        # loss_clip = loss.clip_loss(sims, logit_scale)
        loss_clip2 = loss.clip_loss(sims2, logit_scale2)
        loss_clip3 = loss.clip_loss(sims3, logit_scale3)
        # print(loss_value1)
        # print(loss_value2)
        # loss_value = loss_value2 +0.1*loss_w+ loss_clip
        # sims_gt = sims.detach()
        # mse_loss = calculate_mse_loss(sims2, sims_gt)
        # triplet_loss = get_frame_trip_loss(sims)
        # triplet_loss2 = get_frame_trip_loss(sims2)
        # triplet_loss3 = get_frame_trip_loss(sims3)
        loss_value =  loss_w + loss_clip2    + loss_clip3 + loss_NCE + loss_IVC + loss_mm
        # loss_value =  loss_w + + loss_clip
        sorted_times = None if model.training else get_proposal_results(joint_prob, duration)

        vg_hs_video1 = None if model.training else vg_hs_video
        vg_hs_t1 = None if model.training else vg_hs_t
        vg_hs_v1 = batch_output_tensor
        return loss_value, sorted_times, vg_hs_video1, vg_hs_t1, vg_hs_v1, sentence_number, video_idxs


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

    def on_start(state):
        state['loss_meter'] = AverageMeter()
        state['test_interval'] = int(len(train_dataset)/config.TRAIN.BATCH_SIZE*config.TEST.INTERVAL)
        state['t'] = 1
        model.train()
        if config.VERBOSE:
            state['progress_bar'] = tqdm(total=state['test_interval'])

    def on_forward(state):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        state['loss_meter'].update(state['loss'].item(), 1)

    def on_update(state):# Save All
        if config.VERBOSE:
            state['progress_bar'].update(1)

        if state['t'] % state['test_interval'] == 0:
            model.eval()
            if config.VERBOSE:
                state['progress_bar'].close()

            loss_message = '\niter: {} train loss {:.4f}'.format(state['t'], state['loss_meter'].avg)
            table_message = ''
            if config.TEST.EVAL_TRAIN:
                train_state = engine.test(network, iterator('train_no_shuffle'), 'train')
                train_table = eval.display_results(train_state['Rank@N,mIoU@M'], train_state['miou'],
                                                   'performance on training set')
                table_message += '\n'+ train_table
            if not config.DATASET.NO_VAL:
                val_state = engine.test(network, iterator('val'), 'val')
                state['scheduler'].step(-val_state['loss_meter'].avg)
                loss_message += ' val loss {:.4f}'.format(val_state['loss_meter'].avg)
                val_state['loss_meter'].reset()
                val_table = eval.display_results(val_state['Rank@N,mIoU@M'], val_state['miou'],
                                                 'performance on validation set')
                table_message += '\n'+ val_table

            test_state = engine.test(network, iterator('test'), 'test')
            loss_message += ' test loss {:.4f}'.format(test_state['loss_meter'].avg)
            test_state['loss_meter'].reset()
            test_table = eval.display_results(test_state['Rank@N,mIoU@M'], test_state['miou'],
                                              'performance on testing set')
            table_message += '\n' + test_table

            message = loss_message+table_message+'\n'
            logger.info(message)


            cfg_name = args.cfg
            cfg_name = os.path.basename(cfg_name).split('.yaml')[0]
            saved_model_filename = os.path.join(config.MODEL_DIR, config.DATASET.NAME, cfg_name)
            if not os.path.exists(saved_model_filename):
                print('Make directory %s ...' % saved_model_filename)
                os.makedirs(saved_model_filename)
            saved_model_filename = os.path.join(
                saved_model_filename,
                'iter{:06d}.pkl'.format(state['t'])
            )

            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), saved_model_filename)
            else:
                torch.save(model.state_dict(), saved_model_filename)


            if config.VERBOSE:
                state['progress_bar'] = tqdm(total=state['test_interval'])
            model.train()
            state['loss_meter'].reset()

    def on_end(state):
        if config.VERBOSE:
            state['progress_bar'].close()


    def on_test_start(state):
        state['loss_meter'] = AverageMeter()
        state['sorted_segments_list']=[]
        state['sorted_video_list'] = []
        state['sorted_text_list'] = []
        state['sorted_video_l_list'] = []
        state['sentence_number_list'] = []
        state['sorted_video_id_list'] = []
        if config.VERBOSE:
            if state['split'] == 'test':
                state['progress_bar'] = tqdm(total=math.ceil(len(test_dataset)/config.TEST.BATCH_SIZE))
            else:
                raise NotImplementedError

    def on_test_forward(state):
        if config.VERBOSE:
            state['progress_bar'].update(1)
        state['loss_meter'].update(state['loss'].item(), 1)

        min_idx = min(state['sample']['batch_anno_idxs'])
        batch_indexs = [idx - min_idx for idx in state['sample']['batch_anno_idxs']]
        sorted_segments = [state['output'][i] for i in batch_indexs]
        state['sorted_segments_list'].extend(sorted_segments)
        sorted_videos= state['video'].cpu().detach()
        sorted_texts=state['text'].cpu().detach()
        sorted_videos_l = state['video_l'].cpu().detach()
        sorted_sentence_number=state['sentence_number'].cpu().detach()
        sorted_video_idxs = state['video_idxs']

        # 假设state字典已经被初始化，并且有sorted_video_list和sorted_text_list键
        # state = {'sorted_video_list': [], 'sorted_text_list': []}

        for batch_index in range(sorted_videos.shape[0]):  # 遍历每个批次
            # 从每个批次中提取单个512维向量
            video_vector = sorted_videos[batch_index]
            text_vector = sorted_texts[batch_index]
            video_l_vector=sorted_videos_l[batch_index]
            sorted_sentence_number_vector=sorted_sentence_number[batch_index]
            sorted_video_idxs1= sorted_video_idxs[batch_index]
            # 将这个向量追加到相应的列表中
            state['sorted_video_list'].append(video_vector)
            state['sorted_text_list'].append(text_vector)
            state['sorted_video_l_list'].append(video_l_vector)
            state['sentence_number_list'].append(sorted_sentence_number_vector)
            state['sorted_video_id_list'].append(sorted_video_idxs1)
    def on_test_end(state):
        #############################################################################定位评价指标###################################################################################
        annotations = state['iterator'].dataset.annotations
        state['Rank@N,mIoU@M'], state['miou'], grounding_mask = eval.eval_predictions(state['sorted_segments_list'],
                                                                                      annotations, verbose=False)

        sorted_text_list_all = state['sorted_text_list']
        sorted_video_list_all = state['sorted_video_list']
        sorted_video_layer_list_all = state['sorted_video_l_list']
        sorted_sentence_number_list_all = state['sentence_number_list']
        sorted_video_id_list_all = state['sorted_video_id_list']
        # print(sorted_video_id_list_all)
        #
        sorted_video_layer_list_tensor = torch.stack(sorted_video_layer_list_all, dim=0)
        sorted_text_list_tensor = torch.stack(sorted_text_list_all, dim=0)
        sorted_video_list_tensor = torch.stack(sorted_video_list_all, dim=0)
        sorted_sentence_number_list_tensor = torch.stack(sorted_sentence_number_list_all, dim=0)
        ##############################################################################################################################
        sorted_video_list_tensor_25 = sorted_video_list_tensor#特征
        sorted_video_id_list_all_25 = sorted_video_id_list_all

        logger.info(
            '################################################################################## 使用最后一层图卷积结果的均值进行检索和定位的结果############################################################################')

        rank1_acc_g, rank5_acc_g, rank10_acc_g, rank100_acc_g = calculate_sentence_rank_accuracy_with_grounding_GCN(
            sorted_video_layer_list_tensor, sorted_video_list_tensor_25, sorted_video_id_list_all_25,
            sorted_video_id_list_all,
            sorted_sentence_number_list_tensor, grounding_mask)
        logger.info(f'Rank-1 Accuracy at IOU=0.3, 0.5, 0.7: {rank1_acc_g}')
        logger.info(f'Rank-5 Accuracy at IOU=0.3, 0.5, 0.7: {rank5_acc_g}')
        logger.info(f'Rank-10 Accuracy at IOU=0.3, 0.5, 0.7: {rank10_acc_g}')
        logger.info(f'Rank-100 Accuracy at IOU=0.3, 0.5, 0.7: {rank100_acc_g}')



        if config.VERBOSE:
            state['progress_bar'].close()



    engine = Engine()
    engine.hooks['on_start'] = on_start
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_update'] = on_update
    engine.hooks['on_end'] = on_end
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    engine.train(network,
                 iterator('train'),
                 maxepoch=config.TRAIN.MAX_EPOCH,
                 optimizer=optimizer,
                 scheduler=scheduler)