import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import argparse
import os
import json
from PDFtransformer import TransformerWithPDF, CustomDataset, EnhancedMixedLoss, get_enhanced_lr_scheduler, \
    balance_dataset, plot_confusion_matrix
from torch.amp import autocast, GradScaler  # Use new AMP API
from sklearn.model_selection import StratifiedKFold
import inspect
import random
from tqdm import tqdm
from visualization_utils import plot_cross_validation_summary
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 尝试导入可视化工具（仅新增，不修改原结构）
try:
    from visualization_weights import compute_weights_for_loader, plot_weights_line_chart, plot_weights_mean_bars, \
        plot_decision_level_fixed_weights

    VISUALIZATION_AVAILABLE = True
except Exception as _e:
    VISUALIZATION_AVAILABLE = False
    print(f"visualization_weights.py import failed (visualizations disabled): {_e}")


# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def load_modality_data(data_dir, modality_name):
    feature_path = Path(data_dir) / f"{modality_name}_features6.npy"
    label_path = Path(data_dir) / (f"{modality_name}_labels6"
                                   f".npy")
    # as_posix() 保证路径是 / 分隔
    feature_path = feature_path.as_posix()
    label_path = label_path.as_posix()

    # 添加数据加载日志，查看原始数据分布
    features = np.load(feature_path, allow_pickle=True)
    labels = np.load(label_path, allow_pickle=True)
    print(f"加载 {modality_name} 数据: 形状={features.shape}, 标签分布={np.bincount(labels.flatten())}")

    return features, labels


def preprocess_data(acc_dir, sound_dir, temp_dir, is_training=True):
    """Load and preprocess tri-modality data"""
    acc_features, acc_labels = load_modality_data(acc_dir, "acc")
    sound_features, sound_labels = load_modality_data(sound_dir, "sound")
    temp_features, temp_labels = load_modality_data(temp_dir, "temp")

    # Validate label consistency
    assert np.array_equal(acc_labels, sound_labels) and np.array_equal(acc_labels, temp_labels), "Labels mismatch!"
    return acc_features, sound_features, temp_features, acc_labels


# 修改 balance_dataset 函数，添加更多调试信息
def balance_dataset(acc_data, sound_data, temp_data, labels, sampling_strategy='auto'):
    """平衡数据集，仅用于训练数据"""
    print(f"平衡前的类别分布: {np.bincount(labels.flatten())}")

    # 合并特征用于SMOTE
    combined_features = np.hstack([acc_data, sound_data, temp_data])

    # 应用SMOTE过采样
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    resampled_features, resampled_labels = smote.fit_resample(combined_features, labels)

    print(f"平衡后的类别分布: {np.bincount(resampled_labels)}")

    # 分离不同模态的特征
    acc_dim = acc_data.shape[1]
    sound_dim = sound_data.shape[1]
    temp_dim = temp_data.shape[1]

    balanced_acc = resampled_features[:, :acc_dim]
    balanced_sound = resampled_features[:, acc_dim:acc_dim + sound_dim]
    balanced_temp = resampled_features[:, acc_dim + sound_dim:]

    return balanced_acc, balanced_sound, balanced_temp, resampled_labels


# New function to plot cross-modal attention
def plot_cross_attention(attention_weights, save_path, epoch, fold):
    """
    Visualizes the cross-modal attention weights.
    Averages weights across the batch and heads for simplicity.
    """
    if not attention_weights:
        print("No attention weights found to plot.")
        return

    # Determine the grid size for subplots
    num_plots = len(attention_weights)
    if num_plots == 0: return

    # Try to make a square-ish grid
    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes = axes.flatten()

    fig.suptitle(f'Cross-Modal Attention - Fold {fold}, Epoch {epoch}', fontsize=16)

    for i, (key, value) in enumerate(attention_weights.items()):
        # key is e.g., 'acc_sound', value is the attention tensor
        # value shape: [batch_size, query_len, key_len] after head avg

        # Average across batch
        avg_attn = value.mean(dim=0).cpu().numpy()

        query_modality, key_modality = key.split('_')

        sns.heatmap(avg_attn, ax=axes[i], cmap='viridis')
        axes[i].set_title(f'Query: {query_modality.upper()} -> Key: {key_modality.upper()}')
        axes[i].set_xlabel(f'{key_modality.upper()} Patches (+CLS)')
        axes[i].set_ylabel(f'{query_modality.upper()} Patches (+CLS)')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved cross-attention visualization to {save_path}")


# Improved training function with mixed precision and dynamic mixup
def train_with_mixed_precision(model, train_loader, optimizer, criterion, device, epoch, epochs, scheduler=None,
                               mixup_alpha=0.4, grad_clip_value=1.0):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    # Use new AMP API
    scaler = GradScaler() if device.type == 'cuda' else None

    # Progress bar with tqdm
    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')

    for batch_idx, (acc_data, sound_data, temp_data, labels) in enumerate(pbar):
        acc_data, sound_data, temp_data, labels = acc_data.to(device), sound_data.to(device), temp_data.to(
            device), labels.to(device)

        # Adaptive mixup probability - higher in early epochs, lower in later epochs
        mixup_prob = mixup_alpha * (1 - epoch / epochs * 0.7)  # Maintain some mixup even in later epochs

        # Apply mixup with dynamic probability
        if mixup_alpha > 0 and np.random.random() < mixup_prob:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            idx = torch.randperm(acc_data.size(0)).to(device)
            mixed_acc = lam * acc_data + (1 - lam) * acc_data[idx]
            mixed_sound = lam * sound_data + (1 - lam) * sound_data[idx]
            mixed_temp = lam * temp_data + (1 - lam) * temp_data[idx]

            # Use new AMP API
            with autocast(device_type=device.type, enabled=(scaler is not None)):
                outputs = model(mixed_acc, mixed_sound, mixed_temp)
                squeezed_labels = labels.squeeze()
                squeezed_idx_labels = labels[idx].squeeze()
                loss1 = criterion(outputs, squeezed_labels)
                loss2 = criterion(outputs, squeezed_idx_labels)
                loss = lam * loss1 + (1 - lam) * loss2
        else:
            # Use new AMP API
            with autocast(device_type=device.type, enabled=(scaler is not None)):
                outputs = model(acc_data, sound_data, temp_data)
                labels = labels.squeeze()
                loss = criterion(outputs, labels)

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_value)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_value)
            optimizer.step()

        running_loss += loss.item()

        # Collect predictions
        _, preds = torch.max(outputs, 1)

        # 确保预测和标签都是整数并且是一维数组
        batch_preds = preds.detach().cpu().numpy().astype(int)
        batch_labels = labels.detach().cpu().numpy().astype(int)

        # 确保标签是一维的
        if batch_labels.ndim > 1:
            batch_labels = batch_labels.flatten()

        all_preds.extend(batch_preds)
        all_labels.extend(batch_labels)

        # 调试信息 - 只在第一个epoch的第一批次打印
        if epoch == 0 and batch_idx == 0:
            print(f"Debug - outputs shape: {outputs.shape}")
            print(f"Debug - preds shape: {preds.shape}, dtype: {preds.dtype}")
            print(f"Debug - labels shape: {labels.shape}, dtype: {labels.dtype}")
            print(f"Debug - batch_preds shape: {batch_preds.shape if hasattr(batch_preds, 'shape') else 'list'}")
            print(f"Debug - batch_labels shape: {batch_labels.shape if hasattr(batch_labels, 'shape') else 'list'}")
            print(f"Debug - unique preds: {np.unique(batch_preds)}")
            print(f"Debug - unique labels: {np.unique(batch_labels)}")

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Update learning rate if using batch-level scheduler
        if scheduler is not None and hasattr(scheduler, 'step_batch'):
            scheduler.step()

    # 确保所有数据类型一致
    all_preds = np.array(all_preds, dtype=int)
    all_labels = np.array(all_labels, dtype=int)

    # 确保没有超出范围的预测
    valid_classes = np.unique(all_labels)
    mask = np.isin(all_preds, valid_classes)

    if not np.all(mask):
        # 只在第一个epoch打印警告
        if epoch == 0:
            print(f"Warning: Removing {np.sum(~mask)} predictions with unknown classes")
        all_preds = all_preds[mask]
        all_labels = all_labels[mask]

    # 只在第一个epoch打印最终统计信息
    if epoch == 0:
        print(f"Final train - all_preds shape: {all_preds.shape}, all_labels shape: {all_labels.shape}")
        print(f"Final train - unique preds: {np.unique(all_preds)}")
        print(f"Final train - unique labels: {np.unique(all_labels)}")

    train_metrics = calculate_metrics(all_labels, all_preds)
    train_metrics['loss'] = running_loss / len(train_loader)
    return train_metrics


# Evaluation function
def evaluate(model, data_loader, criterion, device, num_classes):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    running_loss = 0.0

    # 使用静态变量来追踪是否已经打印过调试信息
    if not hasattr(evaluate, "debug_printed"):
        evaluate.debug_printed = False

    with torch.no_grad():
        for batch_idx, (acc_data, sound_data, temp_data, labels) in enumerate(tqdm(data_loader, desc="Evaluating")):
            acc_data, sound_data, temp_data, labels = acc_data.to(device), sound_data.to(device), temp_data.to(
                device), labels.to(device)
            outputs = model(acc_data, sound_data, temp_data)
            labels = labels.squeeze()
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # 确保预测和标签都是整数并且是一维数组
            batch_probs = probs.detach().cpu().numpy()
            batch_preds = preds.detach().cpu().numpy().astype(int)
            batch_labels = labels.detach().cpu().numpy().astype(int)

            # 确保标签是一维的
            if batch_labels.ndim > 1:
                batch_labels = batch_labels.flatten()

            all_probs.append(batch_probs)
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)

            # 调试信息 - 只在第一次评估的第一批次打印
            if not evaluate.debug_printed and batch_idx == 0:
                print(f"Debug eval - outputs shape: {outputs.shape}")
                print(f"Debug eval - preds shape: {preds.shape}, dtype: {preds.dtype}")
                print(f"Debug eval - labels shape: {labels.shape}, dtype: {labels.dtype}")
                print(f"Debug eval - batch_probs shape: {batch_probs.shape}")
                print(
                    f"Debug eval - batch_preds shape: {batch_preds.shape if hasattr(batch_preds, 'shape') else 'list'}")
                print(
                    f"Debug eval - batch_labels shape: {batch_labels.shape if hasattr(batch_labels, 'shape') else 'list'}")
                print(f"Debug eval - unique preds: {np.unique(batch_preds)}")
                print(f"Debug eval - unique labels: {np.unique(batch_labels)}")
                evaluate.debug_printed = True

    # 确保所有数据类型一致
    all_probs = np.vstack(all_probs) if all_probs else None
    all_preds = np.array(all_preds, dtype=int)
    all_labels = np.array(all_labels, dtype=int)

    # 确保没有超出范围的预测
    valid_classes = np.unique(all_labels)
    mask = np.isin(all_preds, valid_classes)

    if not np.all(mask):
        # 只在第一次运行时打印警告
        if not hasattr(evaluate, "warning_printed"):
            print(f"Warning: Removing {np.sum(~mask)} predictions with unknown classes")
            evaluate.warning_printed = True
        all_preds = all_preds[mask]
        all_labels = all_labels[mask]
        if all_probs is not None:
            all_probs = all_probs[mask]

    # 只在第一次评估时打印最终统计信息
    if not hasattr(evaluate, "final_printed"):
        print(f"Final eval - all_preds shape: {all_preds.shape}, all_labels shape: {all_labels.shape}")
        print(f"Final eval - unique preds: {np.unique(all_preds)}")
        print(f"Final eval - unique labels: {np.unique(all_labels)}")
        evaluate.final_printed = True

    metrics = calculate_metrics(all_labels, all_preds, all_probs, num_classes)
    metrics['loss'] = running_loss / len(data_loader)

    try:
        cm = confusion_matrix(all_labels, all_preds)
        metrics['confusion_matrix'] = cm
    except Exception as e:
        if not hasattr(evaluate, "cm_error_printed"):
            print(f"Warning: Could not compute confusion matrix: {e}")
            evaluate.cm_error_printed = True
        metrics['confusion_matrix'] = np.zeros((num_classes, num_classes))

    return metrics


# Calculate metrics function
def calculate_metrics(true_labels, pred_labels, probs=None, num_classes=None):
    # 确保标签类型一致（转换为numpy数组）
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    # 确保标签是一维的
    true_labels = true_labels.flatten()
    pred_labels = pred_labels.flatten()

    # 确保数据类型一致
    true_labels = true_labels.astype(int)
    pred_labels = pred_labels.astype(int)
    metrics = {
        'accuracy': accuracy_score(true_labels, pred_labels),
        'precision': precision_score(true_labels, pred_labels, average='weighted', zero_division=0),
        'recall': recall_score(true_labels, pred_labels, average='weighted', zero_division=0),
        'f1': f1_score(true_labels, pred_labels, average='weighted', zero_division=0),
        'per_class_precision': precision_score(true_labels, pred_labels, average=None, zero_division=0).tolist(),
        'per_class_recall': recall_score(true_labels, pred_labels, average=None, zero_division=0).tolist(),
        'per_class_f1': f1_score(true_labels, pred_labels, average=None, zero_division=0).tolist(),
    }
    if probs is not None and num_classes is not None:
        try:
            # Ensure there are samples from more than one class for AUC calculation
            if len(np.unique(true_labels)) > 1:
                metrics['auc'] = roc_auc_score(true_labels, probs, multi_class='ovr')
            else:
                metrics['auc'] = None
        except Exception:
            metrics['auc'] = None
    return metrics


def enhanced_ensemble_predict(model_paths, acc_data, sound_data, temp_data, device, args, temperature=1.5):
    """改进的集成预测，带温度缩放和置信度加权"""
    all_probs = []
    all_confidences = []

    for model_path in model_paths:
        # 加载模型
        model = TransformerWithPDF(
            input_shape=[acc_data.shape[1], sound_data.shape[1], temp_data.shape[1]],
            num_heads=args.num_heads,
            num_patches=args.num_patches,
            projection_dim=args.projection_dim,
            num_classes=args.num_classes,
            image_input=args.use_image_input
        ).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # 获取预测概率与置信度
        with torch.no_grad():
            acc_tensor = torch.tensor(acc_data, dtype=torch.float32).to(device)
            sound_tensor = torch.tensor(sound_data, dtype=torch.float32).to(device)
            temp_tensor = torch.tensor(temp_data, dtype=torch.float32).to(device)

            outputs = model(acc_tensor, sound_tensor, temp_tensor)
            # 应用温度缩放软化预测
            probs = F.softmax(outputs / temperature, dim=1).cpu().numpy()

            # 计算模型置信度作为预测熵的反函数
            entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
            confidence = 1.0 - entropy / np.log(probs.shape[1])

            all_probs.append(probs)
            all_confidences.append(np.mean(confidence))

    # 按置信度加权模型
    total_confidence = sum(all_confidences)
    if total_confidence > 0:
        weights = np.array([conf / total_confidence for conf in all_confidences])
    else:
        weights = np.ones(len(all_confidences)) / len(all_confidences)

    # 计算加权平均概率
    avg_probs = np.zeros_like(all_probs[0])
    for prob, weight in zip(all_probs, weights):
        avg_probs += prob * weight

    return np.argmax(avg_probs, axis=1), avg_probs


# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--acc_dir', type=str,
                        default="D:/PyCharm/Project_PDFtransformer/University of Ottawa Electric Motor Dataset/acc")
    parser.add_argument('--sound_dir', type=str,
                        default="D:/PyCharm/Project_PDFtransformer/University of Ottawa Electric Motor Dataset/sound")
    parser.add_argument('--temp_dir', type=str,
                        default="D:/PyCharm/Project_PDFtransformer/University of Ottawa Electric Motor Dataset/temp")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_patches', type=int, default=64)
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--save_dir', type=str, default="D:/PyCharm/Project_PDFtransformer/saved6")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_classes', type=int, default=8
                        )
    parser.add_argument('--balance_data', action='store_true', help='Apply data balancing')
    parser.add_argument('--lr_scheduler', type=str, default='cosine_warmup',
                        choices=['step', 'cosine', 'cosine_warm', 'one_cycle', 'plateau', 'cosine_warmup', 'cyclic'])
    parser.add_argument('--focal_gamma', type=float, default=2.5)
    parser.add_argument('--early_stopping', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--aug_strength_start', type=float, default=0.9)
    parser.add_argument('--aug_strength_end', type=float, default=0.4)
    parser.add_argument('--mixup_alpha', type=float, default=0.4)
    parser.add_argument('--label_smoothing', type=float, default=0.15)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=2e-5)
    parser.add_argument('--use_image_input', action='store_true', help='Convert modality data to images')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--dropout_rate', type=float, default=0.4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    args = parser.parse_args()

    # Add SMOTE sampling parameter
    args.use_smote = True

    # Set random seed
    set_seed(args.seed)

    # Create saving directory
    os.makedirs(args.save_dir, exist_ok=True)
    # Create visualization directory
    vis_dir = os.path.join(args.save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess data
    print("Loading and merging all data...")
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

    # Combine datasets
    all_acc = np.vstack((train_acc, val_acc))
    all_sound = np.vstack((train_sound, val_sound))
    all_temp = np.vstack((train_temp, val_temp))
    all_labels = np.concatenate((train_labels.reshape(-1), val_labels.reshape(-1)))

    print(
        f"Combined data shapes - Acceleration: {all_acc.shape}, Sound: {all_sound.shape}, Temperature: {all_temp.shape}")
    print(f"Original class distribution: {np.bincount(all_labels)}")

    # Augmentation strength function
    def get_aug_strength(epoch):
        decay_factor = min(1.0, epoch / (args.epochs * 0.7))
        return max(0.3, args.aug_strength_start * (1 - decay_factor) + args.aug_strength_end * decay_factor)

    # Start cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_scores = []
    cv_auc_scores = []
    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_acc, all_labels)):
        print(f"\n========== Fold {fold + 1}/5 ==========")

        # 分割数据前先打印验证集分布情况
        val_labels_fold = all_labels[val_idx]
        print(f"验证集原始类别分布: {np.bincount(val_labels_fold)}")

        # Split data for current fold
        fold_train_acc, fold_val_acc = all_acc[train_idx], all_acc[val_idx]
        fold_train_sound, fold_val_sound = all_sound[train_idx], all_sound[val_idx]
        fold_train_temp, fold_val_temp = all_temp[train_idx], all_temp[val_idx]
        fold_train_labels, fold_val_labels = all_labels[train_idx], all_labels[val_idx]

        print(f"Fold {fold + 1} split - Train: {len(fold_train_labels)}, Val: {len(fold_val_labels)}")

        # 关键修改：确保验证集数据保持原始分布 - 在训练/验证分割之后再平衡训练数据
        # Balance data using SMOTE if enabled - 只应用于训练数据
        if args.balance_data or args.use_smote:
            print(f"Balancing training data for fold {fold + 1}...")
            fold_train_acc, fold_train_sound, fold_train_temp, fold_train_labels = balance_dataset(
                fold_train_acc, fold_train_sound, fold_train_temp, fold_train_labels,
                sampling_strategy='auto'
            )
            print(f"After balancing - Train: {len(fold_train_labels)}")
            # 验证分割后的标签分布
            print(f"训练集平衡后类别分布: {np.bincount(fold_train_labels)}")
            print(f"验证集类别分布: {np.bincount(fold_val_labels)}")

        # Calculate class weights for current fold
        classes = np.unique(fold_train_labels)
        class_counts = np.bincount(fold_train_labels)
        n_samples = len(fold_train_labels)
        beta = 0.999
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / np.sum(weights) * len(classes)

        class_weights = torch.tensor(weights, dtype=torch.float).to(device)
        print(f"Class weights for fold {fold + 1}: {class_weights}")

        # Create data loaders with enhanced dataset
        initial_aug_strength = get_aug_strength(0)
        print(f"Initial augmentation strength: {initial_aug_strength:.3f}")

        train_dataset = CustomDataset(
            fold_train_acc, fold_train_sound, fold_train_temp, fold_train_labels,
            transform=True,
            aug_strength=initial_aug_strength,
            convert_to_image=args.use_image_input,
            image_size=(args.image_size, args.image_size)
        )

        # 关键修改：使用验证集原始数据，不应用SMOTE
        val_dataset = CustomDataset(
            fold_val_acc, fold_val_sound, fold_val_temp, fold_val_labels,
            transform=False,
            scalers=None if args.use_image_input else train_dataset.scalers,
            convert_to_image=args.use_image_input,
            image_size=(args.image_size, args.image_size)
        )

        if args.balance_data:
            class_sample_count = np.bincount(fold_train_labels)
            weight = 1. / class_sample_count
            samples_weight = torch.tensor([weight[t] for t in fold_train_labels])
            train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                      num_workers=0, pin_memory=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        # Initialize model with improved parameters
        model = TransformerWithPDF(
            input_shape=[fold_train_acc.shape[1], fold_train_sound.shape[1], fold_train_temp.shape[1]],
            num_heads=args.num_heads,
            num_patches=args.num_patches,
            projection_dim=args.projection_dim,
            num_classes=args.num_classes,
            image_input=args.use_image_input,
            dropout_rate=args.dropout_rate
        ).to(device)

        criterion = EnhancedMixedLoss(
            alpha=class_weights,
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

        best_f1 = 0.0
        best_auc = 0.0
        best_combined = 0.0
        no_improve_epochs = 0
        fold_history = {'train': [], 'val': []}

        for epoch in range(args.epochs):
            print(f"\nFold {fold + 1}, Epoch {epoch + 1}/{args.epochs}")

            current_aug_strength = get_aug_strength(epoch)
            train_dataset.aug_strength = current_aug_strength
            print(f"Current augmentation strength: {current_aug_strength:.3f}")

            train_metrics = train_with_mixed_precision(
                model, train_loader, optimizer, criterion, device,
                epoch, args.epochs, scheduler, args.mixup_alpha, args.grad_clip
            )
            fold_history['train'].append(train_metrics)

            val_metrics = evaluate(model, val_loader, criterion, device, args.num_classes)
            fold_history['val'].append(val_metrics)

            # ----- 新增：在每个验证 epoch 完成后生成并保存交叉注意力可视化 -----
            try:
                # The attention weights are automatically captured in model.attention_weights during eval
                if model.attention_weights:
                    attn_save_path = os.path.join(vis_dir, f"fold_{fold + 1}_epoch_{epoch + 1}_cross_attention.png")
                    plot_cross_attention(model.attention_weights, attn_save_path, epoch + 1, fold + 1)
            except Exception as e_vis:
                print(f"Warning: cross-attention visualization failed at fold {fold + 1} epoch {epoch + 1}: {e_vis}")
            # ----- 新增结束 -----

            # ----- 新增（最小改动）：在每个验证 epoch 完成后尝试生成并保存三张权重可视化图片 -----
            if VISUALIZATION_AVAILABLE:
                try:
                    # 1. 确保加载完整验证集数据
                    # 将 max_batches 设置为 None
                    print("Computing weights for full validation set for visualization...")
                    weights_all, _ = compute_weights_for_loader(model, val_loader, device,
                                                                max_batches=None)  # 修改点

                    os.makedirs(vis_dir, exist_ok=True)

                    # 2. 调用新的折线图函数
                    line_chart_path = os.path.join(vis_dir,
                                                   f"fold_{fold + 1}_epoch_{epoch + 1}_val_weights_line_chart.png")

                    if weights_all is not None and weights_all.shape[0] > 0:
                        # 确保样本数量是我们预期的820个左右
                        print(f"Plotting line chart for {weights_all.shape[0]} samples.")
                        plot_weights_line_chart(weights_all, save_path=line_chart_path,
                                                title=f"Fold {fold + 1} Epoch {epoch + 1} - Modal Weights Trend ({weights_all.shape[0]} samples)")
                    else:
                        print("Visualization: no modal weights collected (empty).")

                except Exception as e_vis:
                    print(f"Warning: visualization failed at fold {fold + 1} epoch {epoch + 1}: {e_vis}")
            # ----- 新增结束 -----

            if 'auc' in val_metrics and val_metrics['auc'] is not None:
                print(
                    f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, " +
                    f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}"
                )
                combined_score = 0.7 * val_metrics['f1'] + 0.3 * val_metrics['auc']
            else:
                print(
                    f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, " +
                    f"F1: {val_metrics['f1']:.4f}"
                )
                combined_score = val_metrics['f1']

            if scheduler is not None:
                if args.lr_scheduler == 'plateau':
                    scheduler.step(combined_score)
                elif not hasattr(scheduler, 'step_batch'):
                    scheduler.step()

            if 'confusion_matrix' in val_metrics:
                cm_path = os.path.join(args.save_dir, f"fold_{fold + 1}_cm_epoch_{epoch + 1}.png")
                plot_confusion_matrix(val_metrics['confusion_matrix'],
                                      [f"Class {i}" for i in range(args.num_classes)],
                                      cm_path)

            if combined_score > best_combined:
                best_combined = combined_score
                best_f1 = val_metrics['f1']
                best_auc = val_metrics['auc'] if 'auc' in val_metrics and val_metrics['auc'] is not None else 0.0

                fold_model_path = os.path.join(args.save_dir, f"best_model_fold_{fold + 1}.pth")
                torch.save(model.state_dict(), fold_model_path)
                print(f"Saved new best model for fold {fold + 1}, F1: {best_f1:.4f}, AUC: {best_auc:.4f}")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= args.early_stopping:
                print(f"No improvement for {args.early_stopping} epochs. Early stopping.")
                break

        with open(os.path.join(args.save_dir, f'training_history_fold_{fold + 1}.json'), 'w') as f:
            history_serializable = {}
            for key, value in fold_history.items():
                history_serializable[key] = [{k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                                              for k, v in epoch_metrics.items() if k != 'confusion_matrix'} for
                                             epoch_metrics in value]
            json.dump(history_serializable, f)

        cv_scores.append(best_f1)
        cv_auc_scores.append(best_auc)
        fold_models.append(fold_model_path)

    mean_f1 = np.mean(cv_scores)
    std_f1 = np.std(cv_scores)
    mean_auc = np.mean(cv_auc_scores)
    std_auc = np.std(cv_auc_scores)

    print(f"\nCross-validation complete!")
    print(f"Mean F1 score: {mean_f1:.4f} (±{std_f1:.4f})")
    print(f"Mean AUC score: {mean_auc:.4f} (±{std_auc:.4f})")
    print(f"Per-fold F1 scores: {cv_scores}")
    print(f"Per-fold AUC scores: {cv_auc_scores}")

    cv_results = {
        'mean_f1': float(mean_f1),
        'std_f1': float(std_f1),
        'mean_auc': float(mean_auc),
        'std_auc': float(std_auc),
        'fold_f1_scores': [float(score) for score in cv_scores],
        'fold_auc_scores': [float(score) for score in cv_auc_scores],
        'fold_models': fold_models,
        'args': vars(args)
    }

    with open(os.path.join(args.save_dir, 'cross_validation_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=4)

    # Only create the cross-validation summary visualization
    plot_cross_validation_summary(cv_results, vis_dir)

    # No need to create training curves or other plots


if __name__ == "__main__":
    main()
