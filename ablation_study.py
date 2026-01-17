import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import copy

from PDFtransformer import TransformerWithPDF, CustomDataset, EnhancedMixedLoss, get_enhanced_lr_scheduler
from main3 import preprocess_data, train_with_mixed_precision, evaluate, set_seed, calculate_metrics

# 导入增强版统计分析模块
try:
    from enhanced_ablation_analysis import run_enhanced_analysis

    ENHANCED_ANALYSIS_AVAILABLE = True
except ImportError:
    ENHANCED_ANALYSIS_AVAILABLE = False
    print("警告:  enhanced_ablation_analysis.py 未找到，将跳过增强统计分析")


class AblationTransformerWithPDF(TransformerWithPDF):
    """消融实验专用的Transformer模型，支持禁用特定组件"""

    def __init__(self, input_shape, num_heads, num_patches, projection_dim, num_classes,
                 image_input=False, dropout_rate=0.3,
                 disable_cross_attention=False,
                 disable_prediction_module=False,
                 use_only_modality=None):
        """
        初始化消融实验模型

        参数:
            input_shape:  输入形状
            num_heads:  注意力头数
            num_patches: patch数量
            projection_dim: 投影维度
            num_classes: 类别数
            image_input: 是否使用图像输入
            dropout_rate: dropout比率
            disable_cross_attention: 是否禁用交叉注意力
            disable_prediction_module: 是否禁用预测模块(动态权重)
            use_only_modality: 仅使用单一模态 ('acc', 'sound', 'temp', None)
        """
        super(AblationTransformerWithPDF, self).__init__(
            input_shape, num_heads, num_patches, projection_dim, num_classes,
            image_input, dropout_rate
        )

        self.disable_cross_attention = disable_cross_attention
        self.disable_prediction_module = disable_prediction_module
        self.use_only_modality = use_only_modality

    def forward(self, acc_data, sound_data, temp_data):
        # 单模态模式：将其他模态置零
        if self.use_only_modality == 'acc':
            sound_data = torch.zeros_like(sound_data)
            temp_data = torch.zeros_like(temp_data)
        elif self.use_only_modality == 'sound':
            acc_data = torch.zeros_like(acc_data)
            temp_data = torch.zeros_like(temp_data)
        elif self.use_only_modality == 'temp':
            acc_data = torch.zeros_like(acc_data)
            sound_data = torch.zeros_like(sound_data)

        # 编码各模态
        acc_patches = self.acc_encoder(acc_data)
        sound_patches = self.sound_encoder(sound_data)
        temp_patches = self.temp_encoder(temp_data)

        # 保存原始patches用于残差连接
        acc_original = acc_patches
        sound_original = sound_patches
        temp_original = temp_patches

        # 应用模态特定的transformer层
        for layer in self.acc_transformer:
            acc_patches = layer(acc_patches)
            acc_patches = acc_patches + 0.1 * acc_original

        for layer in self.sound_transformer:
            sound_patches = layer(sound_patches)
            sound_patches = sound_patches + 0.1 * sound_original

        for layer in self.temp_transformer:
            temp_patches = layer(temp_patches)
            temp_patches = temp_patches + 0.1 * temp_original

        # 计算特征用于权重预测和分类
        acc_feat = acc_patches.mean(dim=1)
        sound_feat = sound_patches.mean(dim=1)
        temp_feat = temp_patches.mean(dim=1)

        # 模态特定预测(深度监督)
        acc_pred = self.acc_classifier(acc_feat)
        sound_pred = self.sound_classifier(sound_feat)
        temp_pred = self.temp_classifier(temp_feat)

        # 动态权重计算
        if not self.disable_prediction_module:
            acc_weights = self.acc_prediction_module(acc_feat)
            sound_weights = self.sound_prediction_module(sound_feat)
            temp_weights = self.temp_prediction_module(temp_feat)

            weights = torch.cat([acc_weights, sound_weights, temp_weights], dim=1)
            normalized_weights = torch.nn.functional.softmax(weights, dim=1)
            acc_weights = normalized_weights[:, 0].unsqueeze(1)
            sound_weights = normalized_weights[:, 1].unsqueeze(1)
            temp_weights = normalized_weights[:, 2].unsqueeze(1)
        else:
            # 禁用时使用等权重
            batch_size = acc_patches.size(0)
            acc_weights = torch.ones(batch_size, 1).to(acc_patches.device) / 3
            sound_weights = torch.ones(batch_size, 1).to(acc_patches.device) / 3
            temp_weights = torch.ones(batch_size, 1).to(acc_patches.device) / 3

        # 融合(有无交叉注意力)
        if self.disable_cross_attention:
            fused_features = torch.cat([acc_patches, sound_patches, temp_patches], dim=1)
            seq_len = acc_patches.size(1)
            fused_features = fused_features[:, :seq_len]
        else:
            fused_features = self.fusion_layer(
                acc_patches, sound_patches, temp_patches,
                acc_weights, sound_weights, temp_weights
            )

        # 主transformer处理
        x = fused_features
        mid_features = None

        for i, layer in enumerate(self.transformer_layers):
            x = layer(x)
            if i == len(self.transformer_layers) // 2:
                mid_features = x

        # 中间分类
        mid_logits = self.mid_classifier(mid_features[:, 0]) if mid_features is not None else \
            torch.zeros(x.size(0), self.fc_out[-1].out_features).to(x.device)

        # 主分类
        main_logits = self.fc_out(x[:, 0])

        if not self.training:
            return main_logits
        else:
            final_logits = (
                    main_logits * 0.6 +
                    mid_logits * 0.15 +
                    acc_pred * 0.1 +
                    sound_pred * 0.1 +
                    temp_pred * 0.05
            )
            return final_logits


def run_ablation_experiment_with_cv(config, args, save_dir='ablation_results'):
    """
    运行带5折交叉验证的消融实验

    参数:
        config: 实验配置字典
        args: 命令行参数
        save_dir: 保存目录

    返回:
        交叉验证结果字典
    """
    experiment_name = config['name']
    print(f"\n{'=' * 60}")
    print(f"运行消融实验: {experiment_name}")
    print(f"配置: {config}")
    print(f"{'=' * 60}")

    # 创建实验目录
    exp_dir = os.path.join(save_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 设置随机种子
    set_seed(args.seed)

    # 设备设置
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    print("加载数据...")
    train_acc, train_sound, train_temp, train_labels = preprocess_data(
        os.path.join(args.acc_dir, "train"),
        os.path.join(args.sound_dir, "train"),
        os.path.join(args.temp_dir, "train")
    )

    val_acc, val_sound, val_temp, val_labels = preprocess_data(
        os.path.join(args.acc_dir, "val"),
        os.path.join(args.sound_dir, "val"),
        os.path.join(args.temp_dir, "val")
    )

    # 合并数据集用于交叉验证
    all_acc = np.vstack((train_acc, val_acc))
    all_sound = np.vstack((train_sound, val_sound))
    all_temp = np.vstack((train_temp, val_temp))
    all_labels = np.concatenate((train_labels.reshape(-1), val_labels.reshape(-1)))

    print(f"合并数据形状 - 加速度: {all_acc.shape}, 声音: {all_sound.shape}, 温度: {all_temp.shape}")
    print(f"类别分布: {np.bincount(all_labels)}")

    # 5折分层交叉验证
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_scores = {'accuracy': [], 'f1': [], 'auc': []}
    fold_histories = []
    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_acc, all_labels)):
        print(f"\n{'=' * 40}")
        print(f"第 {fold + 1}/5 折")
        print(f"{'=' * 40}")

        # 分割数据
        fold_train_acc, fold_val_acc = all_acc[train_idx], all_acc[val_idx]
        fold_train_sound, fold_val_sound = all_sound[train_idx], all_sound[val_idx]
        fold_train_temp, fold_val_temp = all_temp[train_idx], all_temp[val_idx]
        fold_train_labels, fold_val_labels = all_labels[train_idx], all_labels[val_idx]

        print(f"训练集:  {len(fold_train_labels)}, 验证集:  {len(fold_val_labels)}")

        # 数据增强配置
        use_augmentation = not config.get('disable_data_augmentation', False)

        train_dataset = CustomDataset(
            fold_train_acc, fold_train_sound, fold_train_temp, fold_train_labels,
            transform=use_augmentation,
            aug_strength=0.5 if use_augmentation else 0.0,
            convert_to_image=args.use_image_input,
            image_size=(args.image_size, args.image_size)
        )

        val_dataset = CustomDataset(
            fold_val_acc, fold_val_sound, fold_val_temp, fold_val_labels,
            transform=False,
            scalers=train_dataset.scalers,
            convert_to_image=args.use_image_input,
            image_size=(args.image_size, args.image_size)
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)

        # 创建消融模型
        model = AblationTransformerWithPDF(
            input_shape=[fold_train_acc.shape[1], fold_train_sound.shape[1], fold_train_temp.shape[1]],
            num_heads=args.num_heads,
            num_patches=args.num_patches,
            projection_dim=args.projection_dim,
            num_classes=args.num_classes,
            image_input=args.use_image_input,
            dropout_rate=args.dropout_rate,
            disable_cross_attention=config.get('disable_cross_attention', False),
            disable_prediction_module=config.get('disable_prediction_module', False),
            use_only_modality=config.get('use_only_modality', None)
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型参数量: {num_params:,}")

        # 损失函数和优化器
        criterion = EnhancedMixedLoss(
            alpha=None,
            gamma=args.focal_gamma,
            smoothing=args.label_smoothing,
            temp=args.temp
        )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        scheduler = get_enhanced_lr_scheduler(optimizer, args, train_loader)

        # 训练循环
        best_f1 = 0.0
        best_auc = 0.0
        best_accuracy = 0.0
        no_improve_epochs = 0
        fold_history = {'train': [], 'val': []}

        for epoch in range(args.epochs):
            print(f"\n第 {fold + 1} 折, Epoch {epoch + 1}/{args.epochs}")

            # 训练
            train_metrics = train_with_mixed_precision(
                model, train_loader, optimizer, criterion, device,
                epoch, args.epochs, scheduler, args.mixup_alpha, args.grad_clip
            )
            fold_history['train'].append(train_metrics)

            # 验证
            val_metrics = evaluate(model, val_loader, criterion, device, args.num_classes)

            # 计算AUC
            val_probs = []
            val_true = []

            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    acc_data, sound_data, temp_data, labels = [b.to(device) for b in batch]
                    outputs = model(acc_data, sound_data, temp_data)
                    probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                    val_probs.append(probs)
                    val_true.append(labels.cpu().numpy())

            val_probs = np.vstack(val_probs)
            val_true = np.concatenate(val_true)

            # 计算多类AUC (one-vs-rest)
            auc_scores = []
            for i in range(args.num_classes):
                if len(np.unique(val_true == i)) > 1:
                    auc_scores.append(roc_auc_score((val_true == i).astype(int), val_probs[:, i]))
                else:
                    auc_scores.append(0.0)

            val_metrics['auc'] = np.mean(auc_scores)
            fold_history['val'].append(val_metrics)

            print(f"训练 Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"验证 Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")

            # 更新学习率
            if scheduler is not None:
                if args.lr_scheduler == 'plateau':
                    scheduler.step(val_metrics['f1'])
                elif not hasattr(scheduler, 'step_batch'):
                    scheduler.step()

            # 保存混淆矩阵
            if 'confusion_matrix' in val_metrics:
                cm_path = os.path.join(exp_dir, f"fold_{fold + 1}_cm_epoch_{epoch + 1}.png")
                plt.figure(figsize=(10, 8))
                sns.heatmap(val_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                            xticklabels=[f"C{i}" for i in range(args.num_classes)],
                            yticklabels=[f"C{i}" for i in range(args.num_classes)])
                plt.xlabel('预测标签')
                plt.ylabel('真实标签')
                plt.title(f'混淆矩阵 - {experiment_name} - 第{fold + 1}折 - Epoch {epoch + 1}')
                plt.tight_layout()
                plt.savefig(cm_path, dpi=300)
                plt.close()

            # 检查是否有改进
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_auc = val_metrics['auc']
                best_accuracy = val_metrics['accuracy']

                fold_model_path = os.path.join(exp_dir, f"best_model_fold_{fold + 1}. pth")
                torch.save(model.state_dict(), fold_model_path)
                print(f"保存最佳模型:  F1={best_f1:.4f}, AUC={best_auc:.4f}")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            # 早停
            if no_improve_epochs >= args.early_stopping:
                print(f"连续{args.early_stopping}个epoch无改进，早停")
                break

        # 保存本折训练历史
        history_path = os.path.join(exp_dir, f'training_history_fold_{fold + 1}. json')
        history_serializable = {}
        for key, value in fold_history.items():
            history_serializable[key] = [
                {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                 for k, v in epoch_metrics.items() if k != 'confusion_matrix'}
                for epoch_metrics in value
            ]
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)

        # 记录本折结果
        cv_scores['f1'].append(best_f1)
        cv_scores['auc'].append(best_auc)
        cv_scores['accuracy'].append(best_accuracy)
        fold_histories.append(fold_history)
        fold_models.append(fold_model_path)

        # 绘制学习曲线
        plot_learning_curves(fold_history, exp_dir, f"{experiment_name}_fold_{fold + 1}")

    # 计算交叉验证统计量
    mean_f1 = np.mean(cv_scores['f1'])
    std_f1 = np.std(cv_scores['f1'])
    mean_auc = np.mean(cv_scores['auc'])
    std_auc = np.std(cv_scores['auc'])
    mean_acc = np.mean(cv_scores['accuracy'])
    std_acc = np.std(cv_scores['accuracy'])

    print(f"\n{'=' * 60}")
    print(f"实验 {experiment_name} 交叉验证完成!")
    print(f"{'=' * 60}")
    print(f"平均 F1: {mean_f1:.4f} (±{std_f1:.4f})")
    print(f"平均 AUC: {mean_auc:.4f} (±{std_auc:. 4f})")
    print(f"平均准确率: {mean_acc:.4f} (±{std_acc:.4f})")
    print(f"各折 F1: {[f'{s:.4f}' for s in cv_scores['f1']]}")
    print(f"各折 AUC:  {[f'{s:.4f}' for s in cv_scores['auc']]}")
    print(f"各折准确率: {[f'{s:.4f}' for s in cv_scores['accuracy']]}")

    # 保存交叉验证结果
    cv_results = {
        'experiment_name': experiment_name,
        'mean_f1': float(mean_f1),
        'std_f1': float(std_f1),
        'mean_auc': float(mean_auc),
        'std_auc': float(std_auc),
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'fold_f1_scores': [float(score) for score in cv_scores['f1']],
        'fold_auc_scores': [float(score) for score in cv_scores['auc']],
        'fold_accuracy_scores': [float(score) for score in cv_scores['accuracy']],
        'fold_models': fold_models,
        'config': config,
        'args': vars(args)
    }

    results_path = os.path.join(exp_dir, 'cross_validation_results. json')
    with open(results_path, 'w') as f:
        json.dump(cv_results, f, indent=4)

    return cv_results


def plot_learning_curves(history, save_dir, experiment_name):
    """绘制学习曲线"""
    plt.figure(figsize=(15, 5))

    epochs = range(1, len(history['train']) + 1)
    train_loss = [m['loss'] for m in history['train']]
    val_loss = [m['loss'] for m in history['val']]
    train_f1 = [m['f1'] for m in history['train']]
    val_f1 = [m['f1'] for m in history['val']]

    # Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-', label='训练Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r-', label='验证Loss', linewidth=2)
    plt.title('损失曲线', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # F1曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_f1, 'b-', label='训练F1', linewidth=2)
    plt.plot(epochs, val_f1, 'r-', label='验证F1', linewidth=2)
    plt.title('F1分数曲线', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('F1', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.suptitle(f'学习曲线 - {experiment_name}', fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'learning_curves_{experiment_name}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()


def create_summary_visualization(ablation_results, save_dir='ablation_results'):
    """创建消融实验汇总可视化"""
    experiment_names = []
    f1_scores = []
    accuracies = []
    aucs = []
    f1_stds = []
    acc_stds = []
    auc_stds = []

    for exp_name, result in ablation_results.items():
        experiment_names.append(exp_name)
        f1_scores.append(result['mean_f1'])
        accuracies.append(result['mean_accuracy'])
        aucs.append(result['mean_auc'])
        f1_stds.append(result['std_f1'])
        acc_stds.append(result['std_accuracy'])
        auc_stds.append(result['std_auc'])

    # 按准确率排序
    sorted_indices = np.argsort(accuracies)[::-1]
    sorted_names = [experiment_names[i] for i in sorted_indices]
    sorted_accuracies = [accuracies[i] for i in sorted_indices]
    sorted_f1_scores = [f1_scores[i] for i in sorted_indices]
    sorted_aucs = [aucs[i] for i in sorted_indices]
    sorted_acc_stds = [acc_stds[i] for i in sorted_indices]
    sorted_f1_stds = [f1_stds[i] for i in sorted_indices]
    sorted_auc_stds = [auc_stds[i] for i in sorted_indices]

    # 创建分组柱状图
    plt.figure(figsize=(16, 8))
    bar_width = 0.25
    index = np.arange(len(sorted_names))

    bars1 = plt.bar(index, sorted_accuracies, bar_width, label='准确率', color='#3274A1',
                    yerr=sorted_acc_stds, capsize=5, edgecolor='black', linewidth=1)
    bars2 = plt.bar(index + bar_width, sorted_f1_scores, bar_width, label='F1分数', color='#E1812C',
                    yerr=sorted_f1_stds, capsize=5, edgecolor='black', linewidth=1)
    bars3 = plt.bar(index + 2 * bar_width, sorted_aucs, bar_width, label='AUC', color='#3A923A',
                    yerr=sorted_auc_stds, capsize=5, edgecolor='black', linewidth=1)

    plt.xlabel('实验配置', fontsize=14)
    plt.ylabel('分数', fontsize=14)
    plt.title('消融实验性能对比 (含标准差误差条)', fontsize=16, fontweight='bold')
    plt.xticks(index + bar_width, sorted_names, rotation=45, ha='right', fontsize=11)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 添加数值标签
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    save_path = os.path.join(save_dir, 'metrics_comparison_bar.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"性能对比柱状图已保存至:  {save_path}")

    # 创建雷达图
    labels = ['准确率', 'F1分数', 'AUC']
    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(12, 10))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], labels, fontsize=12)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
    plt.ylim(0, 1)

    colors = plt.cm.viridis(np.linspace(0, 1, len(experiment_names)))
    for i, name in enumerate(experiment_names):
        values = [
            ablation_results[name]['mean_accuracy'],
            ablation_results[name]['mean_f1'],
            ablation_results[name]['mean_auc']
        ]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=name, color=colors[i])
        ax.fill(angles, values, alpha=0.2, color=colors[i])

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    plt.title('消融实验:  多指标雷达图', fontsize=16, y=1.1, fontweight='bold')
    plt.tight_layout()

    radar_path = os.path.join(save_dir, 'metrics_radar_chart.png')
    plt.savefig(radar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"雷达图已保存至: {radar_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PDFTransformer消融实验')

    # 数据路径
    parser.add_argument('--acc_dir', type=str,
                        default="D:/PyCharm/Project_PDFtransformer/University of Ottawa Electric Motor Dataset/acc")
    parser.add_argument('--sound_dir', type=str,
                        default="D:/PyCharm/Project_PDFtransformer/University of Ottawa Electric Motor Dataset/sound")
    parser.add_argument('--temp_dir', type=str,
                        default="D:/PyCharm/Project_PDFtransformer/University of Ottawa Electric Motor Dataset/temp")

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--early_stopping', type=int, default=70)

    # 模型参数
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_patches', type=int, default=64)
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--dropout_rate', type=float, default=0.4)

    # 训练技巧
    parser.add_argument('--lr_scheduler', type=str, default='cosine_warmup')
    parser.add_argument('--focal_gamma', type=float, default=2.5)
    parser.add_argument('--mixup_alpha', type=float, default=0.4)
    parser.add_argument('--label_smoothing', type=float, default=0.15)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=2e-5)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # 其他参数
    parser.add_argument('--save_dir', type=str, default="D:/PyCharm/Project_PDFtransformer/ablation_results6")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_image_input', action='store_true')
    parser.add_argument('--image_size', type=int, default=224)

    # 统计分析参数
    parser.add_argument('--run_statistical_analysis', action='store_true', default=True,
                        help='是否运行增强版统计分析')

    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 保存实验配置
    config_path = os.path.join(args.save_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"实验配置已保存至: {config_path}")

    # 定义8个消融实验配置
    ablation_configs = [
        {
            'name': 'entire_mechanism',
            'description': '完整模型(所有组件启用)'
        },
        {
            'name': 'no_PredictionModule',
            'description': '禁用动态权重预测模块(使用等权重)',
            'disable_prediction_module': True
        },
        {
            'name': 'no_CustomDataset',
            'description': '禁用数据增强',
            'disable_data_augmentation': True
        },
        {
            'name': 'no_cross_attention',
            'description': '禁用跨模态注意力机制',
            'disable_cross_attention': True
        },
        {
            'name': 'baseline',
            'description': '基线模型(禁用所有增强组件)',
            'disable_prediction_module': True,
            'disable_cross_attention': True,
            'disable_data_augmentation': True
        },
        {
            'name': 'only_acc',
            'description': '仅使用加速度模态',
            'use_only_modality': 'acc'
        },
        {
            'name': 'only_sound',
            'description': '仅使用声音模态',
            'use_only_modality': 'sound'
        },
        {
            'name': 'only_temp',
            'description': '仅使用温度模态',
            'use_only_modality': 'temp'
        }
    ]

    print("\n" + "=" * 70)
    print("                    PDFTransformer 消融实验")
    print("=" * 70)
    print(f"共 {len(ablation_configs)} 个实验配置，每个使用5折交叉验证")
    print("=" * 70 + "\n")

    # 运行所有消融实验
    cv_results = {}
    for i, config in enumerate(ablation_configs):
        print(f"\n>>> 实验 {i + 1}/{len(ablation_configs)}: {config['name']} <<<")
        result = run_ablation_experiment_with_cv(config, args, args.save_dir)
        cv_results[config['name']] = result

    # 保存所有结果
    all_results_path = os.path.join(args.save_dir, 'all_cv_results.json')
    with open(all_results_path, 'w') as f:
        json.dump(cv_results, f, indent=4)
    print(f"\n所有交叉验证结果已保存至: {all_results_path}")

    # 创建基础可视化
    create_summary_visualization(cv_results, args.save_dir)

    # 运行增强版统计分析
    if args.run_statistical_analysis and ENHANCED_ANALYSIS_AVAILABLE:
        print("\n" + "=" * 70)
        print("                运行增强版统计分析")
        print("=" * 70)

        statistical_save_dir = os.path.join(args.save_dir, 'statistical_analysis')
        try:
            report = run_enhanced_analysis(all_results_path, statistical_save_dir)
            print(f"\n统计分析结果已保存至: {statistical_save_dir}")
        except Exception as e:
            print(f"统计分析时出错: {e}")
            import traceback
            traceback.print_exc()
    elif not ENHANCED_ANALYSIS_AVAILABLE:
        print("\n警告:  enhanced_ablation_analysis. py 不可用，跳过增强统计分析")
        print("请确保 enhanced_ablation_analysis.py 在同一目录下")

    # 打印最终排名
    print("\n" + "=" * 70)
    print("              消融实验最终结果 (按F1分数排序)")
    print("=" * 70)

    experiments_by_f1 = sorted(cv_results.items(), key=lambda x: x[1]['mean_f1'], reverse=True)

    print(f"\n{'排名':<4} {'实验名称':<25} {'F1分数':<20} {'准确率':<20} {'AUC':<20}")
    print("-" * 89)

    for i, (exp_name, exp_result) in enumerate(experiments_by_f1):
        f1_str = f"{exp_result['mean_f1']:.4f}±{exp_result['std_f1']:.4f}"
        acc_str = f"{exp_result['mean_accuracy']:.4f}±{exp_result['std_accuracy']:. 4f}"
        auc_str = f"{exp_result['mean_auc']:.4f}±{exp_result['std_auc']:.4f}"
        print(f"{i + 1:<4} {exp_name: <25} {f1_str:<20} {acc_str: <20} {auc_str:<20}")

    print("\n" + "=" * 70)
    print(f"消融实验完成!  所有结果保存于: {args.save_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
