import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from collections import OrderedDict

# 引入项目依赖 (确保 tools, utils, datasets 等文件夹在路径中)
from tools import builder
from utils import misc, config
from utils.logger import *
from pointnet2_ops import pointnet2_utils
from datasets import data_transforms
from torchvision import transforms

# ==========================================
# 1. 辅助函数
# ==========================================

# 对应 runner_finetune.py 中的 test_transforms
test_transforms = transforms.Compose([
    data_transforms.PointcloudScaleAndTranslate(),
])


def strip_module_prefix(state_dict):
    """去除 DDP 训练产生的 'module.' 前缀"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_k = k[7:]  # remove "module."
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict


class FeatureExtractor:
    """
    使用 Hook 机制提取指定层的输入特征
    """

    def __init__(self, module):
        self.features = []
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # input 是一个 tuple，通常第一个元素就是特征张量 (Batch, Feature_Dim)
        # 我们 detach 并转为 cpu numpy
        self.features.append(input[0].detach().cpu().numpy())

    def remove(self):
        self.hook.remove()

    def clear(self):
        self.features = []


# ==========================================
# 2. 核心逻辑
# ==========================================

def get_args():
    parser = argparse.ArgumentParser(description='Generate T-SNE visualization')
    parser.add_argument('--config', type=str, required=True, help='Path to config file (yaml)')
    parser.add_argument('--ckpts', type=str, required=True, help='Path to the finetuned model checkpoint')
    parser.add_argument('--save_path', type=str, default='tsne_visualization.png', help='Path to save the plot')
    parser.add_argument('--perplexity', type=int, default=30, help='T-SNE perplexity parameter')
    parser.add_argument('--max_samples', type=int, default=2000, help='Max samples to visualize to avoid clutter')
    # 添加分布式相关参数占位符，避免 builder 报错
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # 1. 加载配置
    cfg = config.get_config(args)
    cfg.dataset.test.others.bs = 1  # 强制 Batch Size 为 1 (可选，为了稳健)
    args.use_gpu = torch.cuda.is_available()

    logger = get_logger("TSNE_Viz")
    print_log(f"Loading config from {args.config}...", logger=logger)

    # 2. 构建数据集
    print_log("Building dataset...", logger=logger)
    _, test_dataloader = builder.dataset_builder(args, cfg.dataset.test)

    # 3. 构建模型
    print_log("Building model...", logger=logger)
    base_model = builder.model_builder(cfg.model)

    if args.use_gpu:
        base_model.cuda()

    # 4. 加载权重
    print_log(f"Loading checkpoint from {args.ckpts}...", logger=logger)
    checkpoint = torch.load(args.ckpts, map_location='cpu')

    # 处理 checkpoint 键名
    if 'base_model' in checkpoint:
        state_dict = checkpoint['base_model']
    else:
        state_dict = checkpoint

    state_dict = strip_module_prefix(state_dict)

    # 加载参数 (strict=True 确保完全匹配)
    base_model.load_state_dict(state_dict, strict=True)
    base_model.eval()

    # 5. 注册 Hook (关键步骤)
    # 根据你提供的 forward 代码，特征是传给 self.cls_head_finetune 的输入
    # 所以我们要 Hook 这个层
    try:
        extractor = FeatureExtractor(base_model.cls_head_finetune)
        print_log("Successfully registered hook on 'cls_head_finetune'.", logger=logger)
    except AttributeError:
        print_log("Error: Model does not have 'cls_head_finetune'. Check your model definition.", logger=logger)
        return

    # 6. 推理提取特征
    all_features = []
    all_labels = []
    npoints = cfg.npoints

    print_log(f"Extracting features (Max samples: {args.max_samples})...", logger=logger)

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            if len(all_labels) >= args.max_samples:
                break

            points_raw = data[0].cuda()
            label = data[1].cuda()

            # --- 复用 validate 中的预处理逻辑 ---
            # 这里的 FPS 采样逻辑必须与训练时一致，否则特征分布会偏离
            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                point_all = points_raw.size(1)

            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx = pointnet2_utils.furthest_point_sample(points_raw, point_all)
            fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
            points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(), fps_idx).transpose(1,
                                                                                                                  2).contiguous()

            # 缩放平移 (Test Transforms)
            points = test_transforms(points)

            # 前向传播 (Hook 会自动捕获特征)
            _ = base_model(points)

            all_labels.append(label.cpu().numpy())

            if idx % 50 == 0:
                print(f"Processed {idx} batches...")

    # 整合数据
    features_np = np.concatenate(extractor.features, axis=0)  # (N, Feature_Dim)
    labels_np = np.concatenate(all_labels, axis=0).flatten()  # (N,)

    # 移除 Hook
    extractor.remove()

    print_log(f"Features shape: {features_np.shape}", logger=logger)
    print_log(f"Labels shape: {labels_np.shape}", logger=logger)

    # 7. 运行 T-SNE
    print_log("Running T-SNE (this might take a while)...", logger=logger)
    tsne = TSNE(n_components=2, perplexity=args.perplexity, init='pca', random_state=42, learning_rate='auto')
    X_embedded = tsne.fit_transform(features_np)

    # 8. 绘图
    print_log(f"Plotting and saving to {args.save_path}...", logger=logger)
    plot_tsne(X_embedded, labels_np, args.save_path)


def plot_tsne(embeddings, labels, save_path):
    # 设置绘图风格
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("whitegrid")

    plt.figure(figsize=(10, 8))

    # 获取类别数量生成对应色盘
    num_classes = len(np.unique(labels))
    palette = sns.color_palette("tab20", num_classes)  # 使用高对比度色盘

    scatter = sns.scatterplot(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        hue=labels,
        palette=palette,
        legend="full",
        s=60,  # 点的大小
        alpha=0.8,  # 透明度
        edgecolor='w'  # 白色描边
    )

    plt.title('T-SNE Visualization of Finetuned Features', fontsize=18)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # 调整图例位置到外侧，避免遮挡
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Done!")


if __name__ == '__main__':
    main()