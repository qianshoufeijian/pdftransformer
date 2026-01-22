"""
可视化工具：提取并绘制融合前的模态权重。
新增了 plot_weights_line_chart 函数以满足新的可视化需求。
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


# --- 原有函数保持不变 ---
def extract_modal_weights(model, acc, sound, temp, device):
    """
    从模型内部复现到权重预测这一段计算，返回归一化后的每个样本的三模态权重。
    """
    model.eval()
    with torch.no_grad():
        # 转为 tensor
        def _to_tensor(x):
            if isinstance(x, np.ndarray):
                return torch.tensor(x, dtype=torch.float32, device=device)
            elif isinstance(x, torch.Tensor):
                return x.to(device).float()
            else:
                raise ValueError("acc/sound/temp must be numpy or torch.Tensor")

        acc_t = _to_tensor(acc)
        sound_t = _to_tensor(sound)
        temp_t = _to_tensor(temp)

        # 编码
        if model.image_input:
            acc_patches = model.acc_encoder(acc_t)
            sound_patches = model.sound_encoder(sound_t)
            temp_patches = model.temp_encoder(temp_t)
        else:
            acc_patches = model.acc_encoder(acc_t)
            sound_patches = model.sound_encoder(sound_t)
            temp_patches = model.temp_encoder(temp_t)

        acc_original = acc_patches
        sound_original = sound_patches
        temp_original = temp_patches

        # 模态专用 transformer
        for layer in model.acc_transformer:
            acc_patches = layer(acc_patches)
            acc_patches = acc_patches + 0.1 * acc_original
        for layer in model.sound_transformer:
            sound_patches = layer(sound_patches)
            sound_patches = sound_patches + 0.1 * sound_original
        for layer in model.temp_transformer:
            temp_patches = layer(temp_patches)
            temp_patches = temp_patches + 0.1 * temp_original

        # 池化
        acc_feat = acc_patches.mean(dim=1)
        sound_feat = sound_patches.mean(dim=1)
        temp_feat = temp_patches.mean(dim=1)

        # 预测
        acc_score = model.acc_prediction_module(acc_feat)
        sound_score = model.sound_prediction_module(sound_feat)
        temp_score = model.temp_prediction_module(temp_feat)

        # 归一化
        scores = torch.cat([acc_score, sound_score, temp_score], dim=1)
        normalized = F.softmax(scores, dim=1)

        return normalized.cpu().numpy()


def compute_weights_for_loader(model, data_loader, device, max_batches=None):
    """
    对整个 dataloader 批次生成权重集合。
    """
    model.eval()
    weights_list = []
    labels_list = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if max_batches is not None and i >= max_batches:
                break
            if len(batch) == 4:
                acc, sound, temp, labels = batch
                labels_list.append(labels.cpu().numpy().reshape(-1))
            else:
                acc, sound, temp = batch
            weights = extract_modal_weights(model, acc, sound, temp, device)
            weights_list.append(weights)
    weights_all = np.vstack(weights_list) if weights_list else np.zeros((0, 3))
    labels_all = np.concatenate(labels_list) if labels_list else None
    return weights_all, labels_all


# --- 新增的折线图绘制函数 ---
def plot_weights_line_chart(weights, save_path=None, title="Modal Weights Trend"):
    """
    绘制三条折线图，展示各模态权重随样本数量增加的变化。
    - 横轴: 样本数量/索引
    - 左纵轴: 权重值 (0-1)
    - 三条折线分别代表 acc, sound, temp
    """
    if weights is None or weights.size == 0:
        print("Warning: No weights data to plot for line chart.")
        return

    num_samples = weights.shape[0]
    sample_indices = np.arange(num_samples)

    # 创建一个较宽的图像以容纳820个数据点
    plt.figure(figsize=(15, 6))

    # 绘制三条折线
    plt.plot(sample_indices, weights[:, 0], label='acc (加速度)', color='blue', linewidth=1.0, alpha=0.7)
    plt.plot(sample_indices, weights[:, 1], label='sound (声音)', color='green', linewidth=1.0, alpha=0.7)
    plt.plot(sample_indices, weights[:, 2], label='temp (温度)', color='red', linewidth=1.0, alpha=0.7)

    # 设置图表属性
    plt.xlabel("样本索引 (Sample Index)")
    plt.ylabel("模态权重 (Modal Weight)")
    plt.title(title)
    plt.legend()  # 显示图例
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 设置坐标轴范围
    plt.xlim(0, num_samples - 1)
    plt.ylim(0, 1.0)

    # 保存或显示图像
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Line chart saved to {save_path}")
    else:
        plt.show()


def plot_weights_mean_bars(weights, save_path=None, title="Mean Modal Weight"):
    """
    画三模态平均权重的条形图
    """
    if weights is None or weights.size == 0:
        return
    mean_w = weights.mean(axis=0)
    plt.figure(figsize=(5, 4))
    import seaborn as sns
    sns.barplot(x=['acc', 'sound', 'temp'], y=mean_w)
    plt.ylim(0, 1.0)
    plt.ylabel("Mean weight")
    plt.title(title)
    for i, v in enumerate(mean_w):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_weights_by_label(weights, labels, save_dir=None, title_prefix="Modal Weight by Class"):
    """
    按标签统计每个类的平均模态权重并画图。
    weights: (N,3), labels: (N,)
    """
    if weights is None or weights.size == 0 or labels is None:
        return
    classes = np.unique(labels)
    mean_per_class = []
    for c in classes:
        mean_per_class.append(weights[labels == c].mean(axis=0))
    mean_per_class = np.array(mean_per_class)  # (num_classes, 3)

    plt.figure(figsize=(max(6, len(classes) * 0.6), 4))
    for i, m in enumerate(['acc', 'sound', 'temp']):
        plt.plot(classes, mean_per_class[:, i], marker='o', label=m)
    plt.xlabel("Class")
    plt.ylabel("Mean weight")
    plt.title(title_prefix)
    plt.legend()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "weights_by_class.png"), dpi=150)
        plt.close()
    else:
        plt.show()


def plot_decision_level_fixed_weights(save_path=None):
    """
    绘制训练/推理时的固定 logits 加权系数（main, mid, acc_pred, sound_pred, temp_pred）。
    """
    train_w = np.array([0.6, 0.15, 0.1, 0.1, 0.05])
    infer_w = np.array([0.7, 0.1, 0.08, 0.08, 0.04])
    labels = ['main', 'mid', 'acc_pred', 'sound_pred', 'temp_pred']

    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, train_w, width, label='train')
    plt.bar(x + width / 2, infer_w, width, label='inference')
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Weight")
    plt.title("Decision-level fixed weights (train vs inference)")
    plt.legend()
    for i, v in enumerate(train_w):
        plt.text(i - width / 2, v + 0.01, f"{v:.2f}", ha='center')
    for i, v in enumerate(infer_w):
        plt.text(i + width / 2, v + 0.01, f"{v:.2f}", ha='center')
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()