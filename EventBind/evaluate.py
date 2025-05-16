import torch
from torch.utils.data import DataLoader
import os
import json
import numpy as np
from model.EventCLIP import load_clip_to_cpu, EventCLIP
from model.utils.utils import read_yaml 
import torch.nn as nn
from Dataloader.HAIHE.dataset import HAIHE, collate_fn
from tqdm import tqdm
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

def evaluate_one_epoch(model, cfg, dataloader, classnames_num, logit_scale):
    """评估模型在一个 epoch 上的性能并可视化特征空间"""
    epoch_start = time.time()
    total, hit1_v, hit2_v, hit1_ev, hit2_ev = 0, 0, 0, 0, 0
    model = model.eval().float()

    all_logits_tv_v = []
    all_logits_te_e = []
    all_video_features = []
    all_event_features = []
    all_label = []

    # 按类别记录分类正确和样本数
    class_correct_v = defaultdict(int)
    class_total_v = defaultdict(int)
    class_correct_ev = defaultdict(int)
    class_total_ev = defaultdict(int)

    classnames_idxs = np.arange(0, classnames_num)

    print("开始评估...")
    for batch_idx, (frames_features, events, labels, frame_lengths, real_num_frame) in enumerate(tqdm(dataloader, desc="评估进度")):
        if cfg['MODEL']['BACKBONE']['PRE_ENCODING'] == "fp16":
            events = events.float()
            frames_features = frames_features.float()
        with torch.no_grad():
            video_features, event_features, text_features_e, text_features_v \
                = model(events, frames_features, classnames_idxs, frame_lengths, real_num_frame)
            logits_te_e = torch.einsum('bd,bnd->bn', event_features, text_features_e) * logit_scale
            logits_tv_v = torch.einsum('bd,bnd->bn', video_features, text_features_v) * logit_scale
            scores_v = logits_tv_v.softmax(dim=-1)
            scores_ev = logits_te_e.softmax(dim=-1)
            all_logits_tv_v.append(logits_tv_v)
            all_logits_te_e.append(logits_te_e)
            all_video_features.append(video_features)
            all_event_features.append(event_features)

        B, _ = scores_v.size()
        for i in range(B):
            total += 1
            scores_v_i = scores_v[i]
            scores_ev_i = scores_ev[i]
            label_i = labels[i].item()
            all_label.append(label_i)

            # 分类预测
            pred_v = scores_v_i.argmax().item()
            class_total_v[label_i] += 1
            if pred_v == label_i:
                class_correct_v[label_i] += 1
                hit1_v += 1
            if label_i in scores_v_i.topk(2)[1].cpu().detach().numpy():
                hit2_v += 1

            pred_ev = scores_ev_i.argmax().item()
            class_total_ev[label_i] += 1
            if pred_ev == label_i:
                class_correct_ev[label_i] += 1
                hit1_ev += 1
            if label_i in scores_ev_i.topk(2)[1].cpu().detach().numpy():
                hit2_ev += 1

        if (batch_idx + 1) % 10 == 0:
            current_acc1_v = hit1_v / total * 100.
            current_acc1_ev = hit1_ev / total * 100.
            print(f"批次 {batch_idx + 1} / {len(dataloader)}: "
                  f"当前视频模态 acc1_v = {current_acc1_v:.2f}%, "
                  f"当前事件模态 acc1_ev = {current_acc1_ev:.2f}%")

    # 计算整体准确率
    acc1_v = hit1_v / total * 100.
    acc2_v = hit2_v / total * 100.
    acc1_ev = hit1_ev / total * 100.
    acc2_ev = hit2_ev / total * 100.

    # 计算并显示按类别的分类准确率
    print("\n=== 分类准确率（按类别） ===")
    for cls in range(classnames_num):
        acc_v = class_correct_v[cls] / class_total_v[cls] * 100 if class_total_v[cls] > 0 else 0
        acc_ev = class_correct_ev[cls] / class_total_ev[cls] * 100 if class_total_ev[cls] > 0 else 0
        print(f'类别 {cls}: 视频模态 acc1_v = {acc_v:.2f}%, 事件模态 acc1_ev = {acc_ev:.2f}%')

    # 检索性能评估
    all_video_features = torch.cat(all_video_features, dim=0)
    all_event_features = torch.cat(all_event_features, dim=0)
    logits_v_ev = logit_scale * all_video_features @ all_event_features.t()
    scores_v_ev = logits_v_ev.softmax(dim=-1)

    all_logits_tv_v = torch.cat(all_logits_tv_v, dim=0)
    all_logits_te_e = torch.cat(all_logits_te_e, dim=0)
    scores_tv_v = all_logits_tv_v.t().softmax(dim=-1)
    scores_te_e = all_logits_te_e.t().softmax(dim=-1)
    N, n = scores_tv_v.t().size()
    all_label = np.array(all_label)

    # 按类别记录检索成功次数
    class_retrieval_stats_v = {i: {'top1_count': 0, 'top2_count': 0, 'top3_count': 0} for i in range(classnames_num)}
    class_retrieval_stats_e = {i: {'top1_count': 0, 'top2_count': 0, 'top3_count': 0} for i in range(classnames_num)}

    for i in range(n):
        label_i = classnames_idxs[i]
        score_tv_v_i = scores_tv_v[i]
        scores_te_e_i = scores_te_e[i]

        topk_1_v = score_tv_v_i.topk(1)[1].cpu().detach().numpy()[0]
        topk_2_v = score_tv_v_i.topk(2)[1].cpu().detach().numpy()
        topk_3_v = score_tv_v_i.topk(3)[1].cpu().detach().numpy()

        if all_label[topk_1_v] == label_i:
            class_retrieval_stats_v[label_i]['top1_count'] += 1
        if label_i in [all_label[idx] for idx in topk_2_v]:
            class_retrieval_stats_v[label_i]['top2_count'] += 1
        if label_i in [all_label[idx] for idx in topk_3_v]:
            class_retrieval_stats_v[label_i]['top3_count'] += 1

        topk_1_e = scores_te_e_i.topk(1)[1].cpu().detach().numpy()[0]
        topk_2_e = scores_te_e_i.topk(2)[1].cpu().detach().numpy()
        topk_3_e = scores_te_e_i.topk(3)[1].cpu().detach().numpy()

        if all_label[topk_1_e] == label_i:
            class_retrieval_stats_e[label_i]['top1_count'] += 1
        if label_i in [all_label[idx] for idx in topk_2_e]:
            class_retrieval_stats_e[label_i]['top2_count'] += 1
        if label_i in [all_label[idx] for idx in topk_3_e]:
            class_retrieval_stats_e[label_i]['top3_count'] += 1

    # 计算并显示每个类别的检索准确率
    print("\n=== 检索准确率（按类别） ===")
    for cls in range(classnames_num):
        acc_retrieval_1_v_cls = class_retrieval_stats_v[cls]['top1_count'] / 1 * 100.
        acc_retrieval_2_v_cls = class_retrieval_stats_v[cls]['top2_count'] / 1 * 100.
        acc_retrieval_3_v_cls = class_retrieval_stats_v[cls]['top3_count'] / 1 * 100.
        acc_retrieval_1_e_cls = class_retrieval_stats_e[cls]['top1_count'] / 1 * 100.
        acc_retrieval_2_e_cls = class_retrieval_stats_e[cls]['top2_count'] / 1 * 100.
        acc_retrieval_3_e_cls = class_retrieval_stats_e[cls]['top3_count'] / 1 * 100.

        print(f'类别 {cls}:')
        print(f'  文本到视频检索: Top-1={acc_retrieval_1_v_cls:.2f}%, Top-2={acc_retrieval_2_v_cls:.2f}%, Top-3={acc_retrieval_3_v_cls:.2f}%')
        print(f'  文本到事件检索: Top-1={acc_retrieval_1_e_cls:.2f}%, Top-2={acc_retrieval_2_e_cls:.2f}%, Top-3={acc_retrieval_3_e_cls:.2f}%')

    # 视频到事件检索（按类别）
    class_retrieval_stats_v_e = {i: {'top1_count': 0, 'top2_count': 0, 'top3_count': 0, 'total': 0} for i in range(classnames_num)}
    for i in range(N):
        label_i = all_label[i]
        class_retrieval_stats_v_e[label_i]['total'] += 1
        scores_v_ev_i = scores_v_ev[i]
        topk_1_v_e = scores_v_ev_i.topk(1)[1].cpu().detach().numpy()[0]
        topk_2_v_e = scores_v_ev_i.topk(2)[1].cpu().detach().numpy()
        topk_3_v_e = scores_v_ev_i.topk(3)[1].cpu().detach().numpy()

        if all_label[topk_1_v_e] == label_i:
            class_retrieval_stats_v_e[label_i]['top1_count'] += 1
        if label_i in [all_label[idx] for idx in topk_2_v_e]:
            class_retrieval_stats_v_e[label_i]['top2_count'] += 1
        if label_i in [all_label[idx] for idx in topk_3_v_e]:
            class_retrieval_stats_v_e[label_i]['top3_count'] += 1

    print("\n=== 视频到事件检索准确率（按类别） ===")
    for cls in range(classnames_num):
        total = class_retrieval_stats_v_e[cls]['total']
        if total > 0:
            acc_retrieval_1_v_e_cls = class_retrieval_stats_v_e[cls]['top1_count'] / total * 100.
            acc_retrieval_2_v_e_cls = class_retrieval_stats_v_e[cls]['top2_count'] / total * 100.
            acc_retrieval_3_v_e_cls = class_retrieval_stats_v_e[cls]['top3_count'] / total * 100.
        else:
            acc_retrieval_1_v_e_cls = 0
            acc_retrieval_2_v_e_cls = 0
            acc_retrieval_3_v_e_cls = 0
        print(f'类别 {cls}: 视频到事件检索: Top-1={acc_retrieval_1_v_e_cls:.2f}%, Top-2={acc_retrieval_2_v_e_cls:.2f}%, Top-3={acc_retrieval_3_v_e_cls:.2f}%')

    # 整体检索准确率
    acc_retrival_1_e = sum(stats['top1_count'] for stats in class_retrieval_stats_e.values()) / n * 100
    acc_retrival_2_e = sum(stats['top2_count'] for stats in class_retrieval_stats_e.values()) / n * 100
    acc_retrival_3_e = sum(stats['top3_count'] for stats in class_retrieval_stats_e.values()) / n * 100
    acc_retrival_1_v = sum(stats['top1_count'] for stats in class_retrieval_stats_v.values()) / n * 100
    acc_retrival_2_v = sum(stats['top2_count'] for stats in class_retrieval_stats_v.values()) / n * 100
    acc_retrival_3_v = sum(stats['top3_count'] for stats in class_retrieval_stats_v.values()) / n * 100
    acc_retrival_1_v_e = sum(stats['top1_count'] for stats in class_retrieval_stats_v_e.values()) / N * 100
    acc_retrival_2_v_e = sum(stats['top2_count'] for stats in class_retrieval_stats_v_e.values()) / N * 100
    acc_retrival_3_v_e = sum(stats['top3_count'] for stats in class_retrieval_stats_v_e.values()) / N * 100

    # 特征空间可视化
    print("\n开始特征空间可视化...")
    # 合并所有特征
    text_features = text_features_e[0]  # 使用最后一个批次的 text_features_e 作为代表
    all_features = torch.cat([all_video_features, all_event_features, text_features], dim=0)
    # 创建模态标签：0 表示视频，1 表示事件，2 表示文本
    modality_labels = np.concatenate([
        np.zeros(len(all_video_features)),  # 视频特征
        np.ones(len(all_event_features)),   # 事件特征
        np.full(len(text_features), 2)      # 文本特征
    ])
    # 类标签，用于颜色区分
    class_labels = np.concatenate([all_label, all_label, np.arange(classnames_num)])

    # PCA 降维到 2D
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(all_features.cpu().numpy())

    # 可视化
    plt.figure(figsize=(10, 8))
    for modality, color, label in [(0, 'blue', 'Video'), (1, 'green', 'Event'), (2, 'red', 'Text')]:
        mask = modality_labels == modality
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], c=color, label=label, alpha=0.6)
    plt.legend()
    plt.title('Feature Space Visualization (Video, Event, Text)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

    # 保存可视化图
    save_dir = cfg.get('SAVE_DIR', './results')  # 从配置文件获取保存路径，默认为 './results'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'feature_space.png'))
    plt.close()

    # 保存降维后的特征数据
    df = pd.DataFrame(features_2d, columns=['x', 'y'])
    df['modality'] = modality_labels
    df['class_label'] = class_labels
    df.to_csv(os.path.join(save_dir, 'features_2d.csv'), index=False)
    print(f"特征空间可视化已保存至 {save_dir}")

    # 输出评估结果
    print(f"\n评估完成！耗时: {time.time() - epoch_start:.2f} 秒")
    print(f"  视频模态: acc1_v={acc1_v:.2f}%, acc2_v={acc2_v:.2f}%")
    print(f"  事件模态: acc1_ev={acc1_ev:.2f}%, acc2_ev={acc2_ev:.2f}%")
    print(f"  检索指标：")
    print(f"    文本到事件: Top-1={acc_retrival_1_e:.2f}%, Top-2={acc_retrival_2_e:.2f}%, Top-3={acc_retrival_3_e:.2f}%")
    print(f"    文本到视频: Top-1={acc_retrival_1_v:.2f}%, Top-2={acc_retrival_2_v:.2f}%, Top-3={acc_retrival_3_v:.2f}%")
    print(f"    视频到事件: Top-1={acc_retrival_1_v_e:.2f}%, Top-2={acc_retrival_2_v_e:.2f}%, Top-3={acc_retrival_3_v_e:.2f}%")

    return acc1_v, acc2_v, acc1_ev, acc2_ev, \
           acc_retrival_1_e, acc_retrival_2_e, acc_retrival_3_e, \
           acc_retrival_1_v, acc_retrival_2_v, acc_retrival_3_v, \
           acc_retrival_1_v_e, acc_retrival_2_v_e, acc_retrival_3_v_e

if __name__ == '__main__':
    # 1. 读取配置文件
    cfg = read_yaml('/home/username001/nyq/EventBind-master/Configs/HAIHE.yaml')
    
    # 2. 设置设备
    gpus = cfg['Trainer']['GPU_ids']
    device = torch.device("cuda:{}".format(gpus[0]) if torch.cuda.is_available() else "cpu")

    # 3. 加载类别信息
    with open(cfg['Dataset']['Classnames'], "r") as tf:
        classnames_dict = json.load(tf)
    classnames_num = len(classnames_dict)
    print(f"类别数: {classnames_num}")

    # 4. 加载验证数据集
    val_dataset = HAIHE(cfg['Dataset']['Val']['Path'],
                        labels_csv='/home/username001/nyq/EventBind-master/Dataloader/HAIHE/train_labels.csv',  # 请替换为实际验证集的 CSV 文件路径
                        feature_path='/home/username001/nyq/features/val',
                        Labels_file=cfg['Dataset']['Train']['Labels_file'],
                        resize_size=tuple(cfg['Dataset']['resize_size']),
                        representation=cfg['Dataset']['Representation'],
                        augmentation=cfg['Dataset']['Val']['Augmentation'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['Dataset']['Val']['Batch_size'],
                            shuffle=False, drop_last=False, num_workers=4,
                            prefetch_factor=2, pin_memory=True, collate_fn=collate_fn)

    # 5. 加载 CLIP 模型
    print(f"加载 CLIP (backbone: {cfg['MODEL']['BACKBONE']['Name']})")
    clip_model_v = load_clip_to_cpu(cfg)
    clip_model_ev = load_clip_to_cpu(cfg)

    # 6. 初始化 EventCLIP 模型并加载权重
    EventCLIP = EventCLIP(cfg, clip_model_v, clip_model_ev).to(device)
    EventCLIP = nn.DataParallel(EventCLIP, device_ids=gpus, output_device=gpus[0])
    if cfg['MODEL']['Load_Path'] != 'None':
        EventCLIP.load_state_dict(torch.load(cfg['MODEL']['Load_Path'], map_location='cuda:0'), strict=False)
        print(f"已从 {cfg['MODEL']['Load_Path']} 加载模型权重")

    # 7. 获取 logit_scale
    logit_scale = clip_model_v.logit_scale.exp()

    # 8. 执行评估
    acc1_v, acc2_v, acc1_ev, acc2_ev, \
    acc_retrival_1_e, acc_retrival_2_e, acc_retrival_3_e, \
    acc_retrival_1_v, acc_retrival_2_v, acc_retrival_3_v, \
    acc_retrival_1_v_e, acc_retrival_2_v_e, acc_retrival_3_v_e \
        = evaluate_one_epoch(EventCLIP, cfg, val_loader, classnames_num, logit_scale)