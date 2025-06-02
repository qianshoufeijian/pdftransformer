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
import math
from PDFtransformer import TransformerWithPDF, CustomDataset, EnhancedMixedLoss, get_enhanced_lr_scheduler, balance_dataset, plot_confusion_matrix
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
import inspect
import random
from tqdm import tqdm
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from torch.cuda.amp import autocast
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子设置为 {seed}")


def load_modality_data(data_dir, modality_name):
    """加载单模态数据"""
    feature_path = os.path.join(data_dir, f"{modality_name}_features.npy")
    label_path = os.path.join(data_dir, f"{modality_name}_labels.npy")
    return np.load(feature_path, allow_pickle=True), np.load(label_path, allow_pickle=True)


def preprocess_data(acc_dir, sound_dir, temp_dir, is_training=True):
    """加载并预处理三模态数据"""
    # 加载各模态数据
    acc_features, acc_labels = load_modality_data(acc_dir, "acc")
    sound_features, sound_labels = load_modality_data(sound_dir, "sound")
    temp_features, temp_labels = load_modality_data(temp_dir, "temp")

    # 验证标签一致性
    assert np.array_equal(acc_labels, sound_labels) and np.array_equal(acc_labels, temp_labels), "Labels mismatch!"

    return acc_features, sound_features, temp_features, acc_labels


# 改进的训练函数，添加混合精度训练和动态mixup

def train_with_mixed_precision(model, train_loader, optimizer, criterion, device, epoch, epochs, scheduler=None,
                               mixup_alpha=0.4, grad_clip_value=1.0):
    model.train()
    scaler = GradScaler()

    # 初始化缺失变量
    running_loss = 0.0
    all_preds = []
    all_labels = []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    # 动态调整mixup概率
    mixup_prob = mixup_alpha * (1 - min(1.0, epoch / (epochs * 0.7)))

    for batch_idx, (acc_data, sound_data, temp_data, labels) in enumerate(pbar):
        acc_data = acc_data.float().to(device)
        sound_data = sound_data.float().to(device)
        temp_data = temp_data.float().to(device)
        labels = labels.to(device).squeeze()

        # 改进的mixup增强
        if np.random.rand() < mixup_prob:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            idx = torch.randperm(acc_data.size(0)).to(device)

            mixed_acc = lam * acc_data + (1 - lam) * acc_data[idx]
            mixed_sound = lam * sound_data + (1 - lam) * sound_data[idx]
            mixed_temp = lam * temp_data + (1 - lam) * temp_data[idx]

            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(mixed_acc, mixed_sound, mixed_temp)
                loss = criterion(outputs, labels) * lam + criterion(outputs, labels[idx]) * (1 - lam)
        else:
            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(acc_data, sound_data, temp_data)
                loss = criterion(outputs, labels)

        # 梯度裁剪和更新
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        scaler.step(optimizer)
        scaler.update()

        # 动态调整学习率
        if scheduler is not None and hasattr(scheduler, 'step_batch'):
            scheduler.step_batch()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # 更新进度条
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # 汇总训练指标
    train_metrics = {
        'loss': running_loss / len(train_loader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='weighted')
    }
    return train_metrics





# 保留原有训练函数以保持兼容性
def train(model, train_loader, optimizer, criterion, device, epoch, epochs, scheduler=None, mixup_alpha=0.2):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    # 创建梯度缩放器用于混合精度训练
    scaler = torch.amp.GradScaler('cuda')

    # 使用tqdm显示进度条
    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')

    for batch_idx, (acc_data, sound_data, temp_data, labels) in enumerate(pbar):
        acc_data = acc_data.float().to(device)
        sound_data = sound_data.float().to(device)
        temp_data = temp_data.float().to(device)
        labels = labels.to(device).squeeze()

        # 以一定概率应用mixup增强
        if mixup_alpha > 0 and np.random.random() > 0.5:
            # 生成mixup参数
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            idx = torch.randperm(acc_data.size(0)).to(device)

            # 混合数据
            mixed_acc = lam * acc_data + (1 - lam) * acc_data[idx]
            mixed_sound = lam * sound_data + (1 - lam) * sound_data[idx]
            mixed_temp = lam * temp_data + (1 - lam) * temp_data[idx]

            # 使用混合精度
            with torch.amp.autocast('cuda'):
                outputs = model(mixed_acc, mixed_sound, mixed_temp)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(outputs, labels[idx])
                loss = lam * loss1 + (1 - lam) * loss2
        else:
            # 使用混合精度
            with torch.amp.autocast('cuda'):
                outputs = model(acc_data, sound_data, temp_data)
                loss = criterion(outputs, labels)

        # 使用梯度缩放器
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # 反缩放梯度用于裁剪
        scaler.unscale_(optimizer)

        # 梯度裁剪防止梯度爆炸
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # 收集预测结果
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # 更新进度条
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 更新学习率
        if scheduler is not None and hasattr(scheduler, 'step_batch'):
            scheduler.step_batch()

    # 计算训练指标
    train_metrics = calculate_metrics(all_labels, all_preds)
    train_metrics['loss'] = running_loss / len(train_loader)

    return train_metrics


# 改进的评估函数
def evaluate(model, data_loader, criterion, device, num_classes):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for acc_data, sound_data, temp_data, labels in tqdm(data_loader, desc="Evaluating"):
            acc_data = acc_data.float().to(device)
            sound_data = sound_data.float().to(device)
            temp_data = temp_data.float().to(device)
            labels = labels.to(device).squeeze()

            outputs = model(acc_data, sound_data, temp_data)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # 收集预测结果和概率
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_probs.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算评估指标
    all_probs = np.vstack(all_probs) if all_probs else None
    metrics = calculate_metrics(all_labels, all_preds, all_probs, num_classes)
    metrics['loss'] = running_loss / len(data_loader)

    # 生成并保存混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    metrics['confusion_matrix'] = cm

    return metrics


# 修改后的计算指标函数
def calculate_metrics(true_labels, pred_labels, probs=None, num_classes=None):
    metrics = {
        'accuracy': float(accuracy_score(true_labels, pred_labels)),
        'precision': float(precision_score(true_labels, pred_labels, average='weighted', zero_division=0)),
        'recall': float(recall_score(true_labels, pred_labels, average='weighted', zero_division=0)),
        'f1': float(f1_score(true_labels, pred_labels, average='weighted', zero_division=0)),
        'per_class_precision': precision_score(true_labels, pred_labels, average=None, zero_division=0).tolist(),
        'per_class_recall': recall_score(true_labels, pred_labels, average=None, zero_division=0).tolist(),
        'per_class_f1': f1_score(true_labels, pred_labels, average=None, zero_division=0).tolist(),
    }
    # 计算AUC
    if probs is not None and num_classes is not None:
        try:
            # 对于多分类任务，使用 "ovr" (one-vs-rest) 策略计算 AUC
            # 确保 true_labels 是 one-hot 编码格式
            true_labels_one_hot = np.eye(num_classes)[true_labels]
            metrics['auc'] = float(roc_auc_score(true_labels_one_hot, probs, multi_class='ovr', average='weighted'))
        except Exception as e:
            metrics['auc'] = None
    return metrics


# 增强的模型集成预测函数
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
    weights = np.array([conf / total_confidence for conf in all_confidences])

    # 计算加权平均概率
    avg_probs = np.zeros_like(all_probs[0])
    for prob, weight in zip(all_probs, weights):
        avg_probs += prob * weight

    return np.argmax(avg_probs, axis=1), avg_probs

def load_calibration_layers(model_paths, device, num_classes):
    calibration_layers = []
    for model_path in model_paths:
        # 提取 fold 编号
        fold_num = int(os.path.basename(model_path).split('_')[-1].split('.')[0])
        calib_path = os.path.join(os.path.dirname(model_path), f"calibration_layer_fold_{fold_num}.pth")

        # 加载校准层
        if os.path.exists(calib_path):
            calib_layer = nn.Linear(num_classes, num_classes)
            calib_layer.load_state_dict(torch.load(calib_path, map_location=device))
            calibration_layers.append(calib_layer.to(device))
        else:
            print(f"Warning: Calibration layer not found for fold {fold_num}. Skipping.")
            calibration_layers.append(None)
    return calibration_layers


# 保留原有集成预测函数以保持兼容性
def ensemble_predict(model_paths, acc_data, sound_data, temp_data, device, args=None):
    """使用所有折训练的模型进行集成预测"""
    all_probs = []

    for model_path in model_paths:
        # 加载模型
        if args is None:
            # 提取参数值
            model_filename = os.path.basename(model_path)
            model_parts = model_filename.split('_')
            # 假设模型命名包含参数信息
            num_heads = 8
            num_patches = 64
            projection_dim = 256
            num_classes = 10
            image_input = False
        else:
            num_heads = args.num_heads
            num_patches = args.num_patches
            projection_dim = args.projection_dim
            num_classes = args.num_classes
            image_input = args.use_image_input if hasattr(args, 'use_image_input') else False

        model = TransformerWithPDF(
            input_shape=[acc_data.shape[1], sound_data.shape[1], temp_data.shape[1]],
            num_heads=num_heads,
            num_patches=num_patches,
            projection_dim=projection_dim,
            num_classes=num_classes,
            image_input=image_input
        ).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # 获取预测概率
        with torch.no_grad():
            acc_tensor = torch.tensor(acc_data, dtype=torch.float32).to(device)
            sound_tensor = torch.tensor(sound_data, dtype=torch.float32).to(device)
            temp_tensor = torch.tensor(temp_data, dtype=torch.float32).to(device)

            outputs = model(acc_tensor, sound_tensor, temp_tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)

    # 平均所有模型的预测概率
    avg_probs = np.mean(all_probs, axis=0)
    return np.argmax(avg_probs, axis=1)


# 改进的主函数
def main():
    # Parameter setup
    parser = argparse.ArgumentParser()
    parser.add_argument('--acc_dir', type=str,
                        default="D:/PyCharm/PythonProject1/PDF-main/University of Ottawa Electric Motor Dataset/acc")
    parser.add_argument('--sound_dir', type=str,
                        default="D:/PyCharm/PythonProject1/PDF-main/University of Ottawa Electric Motor Dataset/sound")
    parser.add_argument('--temp_dir', type=str,
                        default="D:/PyCharm/PythonProject1/PDF-main/University of Ottawa Electric Motor Dataset/temp")
    parser.add_argument('--batch_size', type=int, default=32)  # Reduced for stability
    parser.add_argument('--epochs', type=int, default=120)  # Increased epochs
    parser.add_argument('--lr', type=float, default=3e-4)  # Adjusted learning rate
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_patches', type=int, default=32)  # Reduced patches
    parser.add_argument('--projection_dim', type=int, default=128)  # Adjusted dimension
    parser.add_argument('--save_dir', type=str, default=r"D:\PyCharm\PythonProject1\PDF-main\saved")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--balance_data', action='store_true', help='Apply data balancing')
    parser.add_argument('--lr_scheduler', type=str, default='cosine_warmup',
                        choices=['step', 'cosine', 'cosine_warm', 'one_cycle', 'plateau', 'cosine_warmup', 'cyclic'])
    parser.add_argument('--focal_gamma', type=float, default=1.5)
    parser.add_argument('--early_stopping', type=int, default=20)  # Increased patience
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--aug_strength_start', type=float, default=0.85)
    parser.add_argument('--aug_strength_end', type=float, default=0.4)
    parser.add_argument('--mixup_alpha', type=float, default=0.3)  # Adjusted mixup
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--temp', type=float, default=0.8)  # Reduced temperature
    parser.add_argument('--weight_decay', type=float, default=2e-5)  # Increased weight decay
    parser.add_argument('--use_image_input', action='store_true', help='Convert modality data to images')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--dropout_rate', type=float, default=0.4)  # New parameter for dropout rate
    parser.add_argument('--grad_clip', type=float, default=1.0)  # New parameter for gradient clipping
    args = parser.parse_args()

    # Add SMOTE sampling parameter
    args.use_smote = True  # Always use SMOTE for better handling of imbalance

    # Set random seed
    set_seed(args.seed)

    # Create saving directory
    os.makedirs(args.save_dir, exist_ok=True)

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

    # Adaptive augmentation strength function
    def get_aug_strength(epoch):
        # Linear decay with a floor
        decay_factor = min(1.0, epoch / (args.epochs * 0.7))
        return max(0.3, args.aug_strength_start * (1 - decay_factor) + args.aug_strength_end * decay_factor)

    # Start cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_scores = []
    cv_auc_scores = []
    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_acc, all_labels)):
        print(f"\n========== Fold {fold + 1}/5 ==========")

        # Split data for current fold
        fold_train_acc, fold_val_acc = all_acc[train_idx], all_acc[val_idx]
        fold_train_sound, fold_val_sound = all_sound[train_idx], all_sound[val_idx]
        fold_train_temp, fold_val_temp = all_temp[train_idx], all_temp[val_idx]
        fold_train_labels, fold_val_labels = all_labels[train_idx], all_labels[val_idx]

        print(f"Fold {fold + 1} split - Train: {len(fold_train_labels)}, Val: {len(fold_val_labels)}")

        # Balance data using SMOTE if enabled
        if args.balance_data or args.use_smote:
            print(f"Balancing training data for fold {fold + 1}...")
            fold_train_acc, fold_train_sound, fold_train_temp, fold_train_labels = balance_dataset(
                fold_train_acc, fold_train_sound, fold_train_temp, fold_train_labels,
                sampling_strategy='auto'  # Use auto strategy for SMOTE
            )
            print(f"After balancing - Train: {len(fold_train_labels)}")

        # Calculate class weights for current fold
        classes = np.unique(fold_train_labels)
        class_counts = np.bincount(fold_train_labels)
        n_samples = len(fold_train_labels)

        # Calculate class weights using effective samples formula
        beta = 0.999  # Smoother weight distribution
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / np.array(effective_num)
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
            image_size=(args.image_size, args.image_size),
            apply_denoise=True,  # 确保在训练集上应用滤波
            #filter_type="wavelet"  # 根据需求调整，这里示例使用 "all"，即使用小波+低通+高通
        )

        val_dataset = CustomDataset(
            fold_val_acc, fold_val_sound, fold_val_temp, fold_val_labels,
            transform=False,
            scalers=None if args.use_image_input else train_dataset.scalers,  # 验证集使用训练集的scalers
            convert_to_image=args.use_image_input,
            image_size=(args.image_size, args.image_size),
            apply_denoise=True,  # 确保在验证集上应用滤波
            #filter_type="wavelet"  # 根据需求调整，这里示例使用 "all"，即使用小波+低通+高通
        )

        # Use weighted sampling for training
        if args.balance_data:
            class_sample_count = np.bincount(fold_train_labels)
            weight = 1. / class_sample_count
            samples_weight = torch.tensor([weight[t] for t in fold_train_labels])
            train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                      num_workers=4, pin_memory=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=4, pin_memory=True)

        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)

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

        # Loss function and optimizer
        criterion = EnhancedMixedLoss(
            alpha=class_weights,
            gamma=args.focal_gamma,
            smoothing=args.label_smoothing,
            temp=args.temp
        )

        # Use AdamW with weight decay
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),  # Default betas
            eps=1e-8
        )

        # Improved scheduler
        scheduler = get_enhanced_lr_scheduler(optimizer, args, train_loader)

        # Training loop
        best_f1 = 0.0
        best_auc = 0.0
        best_combined = 0.0
        no_improve_epochs = 0
        fold_history = {'train': [], 'val': []}

        for epoch in range(args.epochs):
            print(f"\nFold {fold + 1}, Epoch {epoch + 1}/{args.epochs}")

            # Update augmentation strength
            current_aug_strength = get_aug_strength(epoch)
            train_dataset.aug_strength = current_aug_strength
            print(f"Current augmentation strength: {current_aug_strength:.3f}")

            # Train with improved mixed precision and gradient handling
            train_metrics = train_with_mixed_precision(
                model, train_loader, optimizer, criterion, device,
                epoch, args.epochs, scheduler, args.mixup_alpha, args.grad_clip
            )
            fold_history['train'].append(train_metrics)

            # Evaluate
            val_metrics = evaluate(model, val_loader, criterion, device, args.num_classes)
            fold_history['val'].append(val_metrics)

            # Output validation metrics
            if 'auc' in val_metrics and val_metrics['auc'] is not None:
                print(
                    f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, " +
                    f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}"
                )
                # Calculate combined metric with emphasis on F1
                combined_score = 0.7 * val_metrics['f1'] + 0.3 * val_metrics['auc']
            else:
                print(
                    f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, " +
                    f"F1: {val_metrics['f1']:.4f}"
                )
                combined_score = val_metrics['f1']

            # Update learning rate
            if scheduler is not None:
                if args.lr_scheduler == 'plateau':
                    scheduler.step(combined_score)
                elif not hasattr(scheduler, 'step_batch'):
                    scheduler.step()

            # Save confusion matrix
            if 'confusion_matrix' in val_metrics:
                cm_path = os.path.join(args.save_dir, f"fold_{fold + 1}_cm_epoch_{epoch + 1}.png")
                plot_confusion_matrix(val_metrics['confusion_matrix'],
                                      [f"Class {i}" for i in range(args.num_classes)],
                                      cm_path)

            # Save best model
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

            # Early stopping
            if no_improve_epochs >= args.early_stopping:
                print(f"No improvement for {args.early_stopping} epochs. Early stopping.")
                break

        # Save training history
        with open(os.path.join(args.save_dir, f'training_history_fold_{fold + 1}.json'), 'w') as f:
            history_serializable = {}
            for key, value in fold_history.items():
                history_serializable[key] = [{k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                                              for k, v in epoch_metrics.items()
                                              if k != 'confusion_matrix'}
                                             for epoch_metrics in value]
            json.dump(history_serializable, f)

        # Record fold scores
        cv_scores.append(best_f1)
        cv_auc_scores.append(best_auc)
        fold_models.append(fold_model_path)

    # Calculate and output cross-validation performance
    mean_f1 = np.mean(cv_scores)
    std_f1 = np.std(cv_scores)
    mean_auc = np.mean(cv_auc_scores)
    std_auc = np.std(cv_auc_scores)

    print(f"\nCross-validation complete!")
    print(f"Mean F1 score: {mean_f1:.4f} (±{std_f1:.4f})")
    print(f"Mean AUC score: {mean_auc:.4f} (±{std_auc:.4f})")
    print(f"Per-fold F1 scores: {cv_scores}")
    print(f"Per-fold AUC scores: {cv_auc_scores}")

    # 保存交叉验证结果
    cv_results = {
        'mean_f1': float(mean_f1),
        'std_f1': float(std_f1),
        'mean_auc': float(mean_auc),
        'std_auc': float(std_auc),
        'fold_f1_scores': [float(f1) for f1 in cv_scores],
        'fold_auc_scores': [float(auc) for auc in cv_auc_scores],
        'fold_models': fold_models,
        'args': vars(args)  # 保存所有参数设置
    }
    with open(os.path.join(args.save_dir, 'cross_validation_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=4)

    # 保存增强的集成预测函数
    with open(os.path.join(args.save_dir, 'enhanced_ensemble_predict.py'), 'w') as f:
        f.write(inspect.getsource(enhanced_ensemble_predict))

    print("交叉验证和模型集成设置完成!")


if __name__ == "__main__":
    main()