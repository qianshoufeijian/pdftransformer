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
from main import preprocess_data, train_with_mixed_precision, evaluate, set_seed, calculate_metrics


class AblationTransformerWithPDF(TransformerWithPDF):
    def __init__(self, input_shape, num_heads, num_patches, projection_dim, num_classes,
                 image_input=False, dropout_rate=0.3,
                 # Ablation parameters
                 disable_cross_attention=False,
                 disable_prediction_module=False,
                 use_only_modality=None):  # None, 'acc', 'sound', 'temp'

        # Initialize parent class
        super(AblationTransformerWithPDF, self).__init__(
            input_shape, num_heads, num_patches, projection_dim, num_classes,
            image_input, dropout_rate
        )

        # Save ablation parameters
        self.disable_cross_attention = disable_cross_attention
        self.disable_prediction_module = disable_prediction_module
        self.use_only_modality = use_only_modality

    def forward(self, acc_data, sound_data, temp_data):
        # Single modality mode
        if self.use_only_modality == 'acc':
            sound_data = torch.zeros_like(sound_data)
            temp_data = torch.zeros_like(temp_data)
        elif self.use_only_modality == 'sound':
            acc_data = torch.zeros_like(acc_data)
            temp_data = torch.zeros_like(temp_data)
        elif self.use_only_modality == 'temp':
            acc_data = torch.zeros_like(acc_data)
            sound_data = torch.zeros_like(sound_data)

        # Encode each modality
        if self.image_input:
            acc_patches = self.acc_encoder(acc_data)
            sound_patches = self.sound_encoder(sound_data)
            temp_patches = self.temp_encoder(temp_data)
        else:
            acc_patches = self.acc_encoder(acc_data)
            sound_patches = self.sound_encoder(sound_data)
            temp_patches = self.temp_encoder(temp_data)

        # Save original patches for residual connection
        acc_original = acc_patches
        sound_original = sound_patches
        temp_original = temp_patches

        # Apply modality-specific transformers with residual connections
        for layer in self.acc_transformer:
            acc_patches = layer(acc_patches)
            acc_patches = acc_patches + 0.1 * acc_original

        for layer in self.sound_transformer:
            sound_patches = layer(sound_patches)
            sound_patches = sound_patches + 0.1 * sound_original

        for layer in self.temp_transformer:
            temp_patches = layer(temp_patches)
            temp_patches = temp_patches + 0.1 * temp_original

        # Calculate features for weight prediction and classification
        acc_feat = acc_patches.mean(dim=1)
        sound_feat = sound_patches.mean(dim=1)
        temp_feat = temp_patches.mean(dim=1)

        # Modality-specific predictions for deep supervision
        acc_pred = self.acc_classifier(acc_feat)
        sound_pred = self.sound_classifier(sound_feat)
        temp_pred = self.temp_classifier(temp_feat)

        # Dynamic weight calculation (if enabled)
        if not self.disable_prediction_module:
            acc_weights = self.acc_prediction_module(acc_feat)
            sound_weights = self.sound_prediction_module(sound_feat)
            temp_weights = self.temp_prediction_module(temp_feat)

            # Weight normalization with softmax
            weights = torch.cat([acc_weights, sound_weights, temp_weights], dim=1)
            normalized_weights = torch.nn.functional.softmax(weights, dim=1)
            acc_weights = normalized_weights[:, 0].unsqueeze(1)
            sound_weights = normalized_weights[:, 1].unsqueeze(1)
            temp_weights = normalized_weights[:, 2].unsqueeze(1)
        else:
            # Equal weights if disabled
            batch_size = acc_patches.size(0)
            acc_weights = torch.ones(batch_size, 1).to(acc_patches.device) / 3
            sound_weights = torch.ones(batch_size, 1).to(acc_patches.device) / 3
            temp_weights = torch.ones(batch_size, 1).to(acc_patches.device) / 3

        # Apply fusion with or without cross-attention
        if self.disable_cross_attention:
            # When cross-attention is disabled, we concatenate the features directly
            fused_features = torch.cat([acc_patches, sound_patches, temp_patches], dim=1)
            # Apply a simple linear layer to get back to original sequence length
            seq_len = acc_patches.size(1)
            fused_features = fused_features[:, :seq_len]
        else:
            # Normal operation with cross-attention
            fused_features = self.fusion_layer(
                acc_patches, sound_patches, temp_patches,
                acc_weights, sound_weights, temp_weights
            )

        # Main transformer processing
        x = fused_features
        mid_features = None
        secondary_features = None

        for i, layer in enumerate(self.transformer_layers):
            x = layer(x)
            if i == len(self.transformer_layers) // 2:
                mid_features = x
            elif i == len(self.transformer_layers) // 4:
                secondary_features = x

        # Intermediate classification (if available)
        mid_logits = self.mid_classifier(mid_features[:, 0]) if mid_features is not None else \
            torch.zeros(x.size(0), self.fc_out[-1].out_features).to(x.device)

        # Main classification from first token
        main_logits = self.fc_out(x[:, 0])

        if not self.training:
            # During inference, just use main logits
            return main_logits
        else:
            # Combine predictions with proper weights during training
            final_logits = (
                    main_logits * 0.6 +
                    mid_logits * 0.15 +
                    acc_pred * 0.1 +
                    sound_pred * 0.1 +
                    temp_pred * 0.05
            )
            return final_logits


def run_ablation_experiment_with_cv(config, args, save_dir='ablation_results'):
    """Run an ablation experiment with 5-fold cross-validation"""
    experiment_name = config['name']
    print(f"\n{'=' * 50}")
    print(f"Running ablation experiment: {experiment_name}")
    print(f"{'=' * 50}")

    # Create directory to save results
    exp_dir = os.path.join(save_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Set random seed
    set_seed(args.seed)

    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess data
    print("Loading data...")
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

    # Combine datasets for cross-validation
    all_acc = np.vstack((train_acc, val_acc))
    all_sound = np.vstack((train_sound, val_sound))
    all_temp = np.vstack((train_temp, val_temp))
    all_labels = np.concatenate((train_labels.reshape(-1), val_labels.reshape(-1)))

    print(
        f"Combined data shapes - Acceleration: {all_acc.shape}, Sound: {all_sound.shape}, Temperature: {all_temp.shape}")
    print(f"Combined class distribution: {np.bincount(all_labels)}")

    # Start 5-fold cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_scores = {'accuracy': [], 'f1': [], 'auc': []}
    fold_histories = []
    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_acc, all_labels)):
        print(f"\n========== Fold {fold + 1}/5 ==========")

        # Split data for current fold
        fold_train_acc, fold_val_acc = all_acc[train_idx], all_acc[val_idx]
        fold_train_sound, fold_val_sound = all_sound[train_idx], all_sound[val_idx]
        fold_train_temp, fold_val_temp = all_temp[train_idx], all_temp[val_idx]
        fold_train_labels, fold_val_labels = all_labels[train_idx], all_labels[val_idx]

        print(f"Fold {fold + 1} split - Train: {len(fold_train_labels)}, Val: {len(fold_val_labels)}")

        # For no_CustomDataset experiment, disable data augmentation
        use_augmentation = not config.get('disable_data_augmentation', False)

        train_dataset = CustomDataset(
            fold_train_acc, fold_train_sound, fold_train_temp, fold_train_labels,
            transform=use_augmentation,  # Only use augmentation if not disabled
            aug_strength=0.5 if use_augmentation else 0.0,  # Set aug_strength to 0 if disabled
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

        # Create model with ablation configuration
        model = AblationTransformerWithPDF(
            input_shape=[fold_train_acc.shape[1], fold_train_sound.shape[1], fold_train_temp.shape[1]],
            num_heads=args.num_heads,
            num_patches=args.num_patches,
            projection_dim=args.projection_dim,
            num_classes=args.num_classes,
            image_input=args.use_image_input,
            dropout_rate=args.dropout_rate,
            # Ablation parameters from config
            disable_cross_attention=config.get('disable_cross_attention', False),
            disable_prediction_module=config.get('disable_prediction_module', False),
            use_only_modality=config.get('use_only_modality', None)
        ).to(device)

        # Count trainable parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model has {num_params:,} trainable parameters")

        # Setup loss and optimizer
        criterion = EnhancedMixedLoss(
            alpha=None,  # No class weights for ablation studies to keep things simple
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

        # Training loop
        best_f1 = 0.0
        best_auc = 0.0
        best_accuracy = 0.0
        no_improve_epochs = 0
        fold_history = {'train': [], 'val': []}

        for epoch in range(args.epochs):
            print(f"\nFold {fold + 1}, Epoch {epoch + 1}/{args.epochs}")

            # Train
            train_metrics = train_with_mixed_precision(
                model, train_loader, optimizer, criterion, device,
                epoch, args.epochs, scheduler, args.mixup_alpha, args.grad_clip
            )
            fold_history['train'].append(train_metrics)

            # Validate
            val_metrics = evaluate(model, val_loader, criterion, device, args.num_classes)

            # Calculate AUC for multi-class classification (one-vs-rest)
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

            # Calculate one-vs-rest AUC for each class
            auc_scores = []
            for i in range(args.num_classes):
                if len(np.unique(val_true == i)) > 1:  # Only calculate AUC if both classes present
                    auc_scores.append(roc_auc_score((val_true == i).astype(int), val_probs[:, i]))
                else:
                    auc_scores.append(0.0)

            # Macro AUC (average across classes)
            val_metrics['auc'] = np.mean(auc_scores)

            fold_history['val'].append(val_metrics)

            print(f"Train Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")

            # Update scheduler if needed
            if scheduler is not None and args.lr_scheduler == 'plateau':
                scheduler.step(val_metrics['f1'])
            elif scheduler is not None and not hasattr(scheduler, 'step_batch'):
                scheduler.step()

            # Save confusion matrix
            if 'confusion_matrix' in val_metrics:
                cm_path = os.path.join(exp_dir, f"fold_{fold + 1}_cm_epoch_{epoch + 1}.png")
                plt.figure(figsize=(10, 8))
                sns.heatmap(val_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                            xticklabels=[f"C{i}" for i in range(args.num_classes)],
                            yticklabels=[f"C{i}" for i in range(args.num_classes)])
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Confusion Matrix - {experiment_name} - Fold {fold + 1} - Epoch {epoch + 1}')
                plt.tight_layout()
                plt.savefig(cm_path, dpi=300)
                plt.close()

            # Check for improvement
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_auc = val_metrics['auc']
                best_accuracy = val_metrics['accuracy']

                fold_model_path = os.path.join(exp_dir, f"best_model_fold_{fold + 1}.pth")
                torch.save(model.state_dict(), fold_model_path)
                print(f"Saved new best model for fold {fold + 1}, F1: {best_f1:.4f}, AUC: {best_auc:.4f}")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            # Early stopping
            if no_improve_epochs >= args.early_stopping:
                print(f"No improvement for {args.early_stopping} epochs. Early stopping.")
                break

        # Save fold history
        with open(os.path.join(exp_dir, f'training_history_fold_{fold + 1}.json'), 'w') as f:
            history_serializable = {}
            for key, value in fold_history.items():
                history_serializable[key] = [{k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                                              for k, v in epoch_metrics.items() if k != 'confusion_matrix'} for
                                             epoch_metrics in value]
            json.dump(history_serializable, f)

        # Save fold results
        cv_scores['f1'].append(best_f1)
        cv_scores['auc'].append(best_auc)
        cv_scores['accuracy'].append(best_accuracy)
        fold_histories.append(fold_history)
        fold_models.append(fold_model_path)

        # Plot learning curves for this fold
        plot_learning_curves(fold_history, exp_dir, f"{experiment_name}_fold_{fold + 1}")

    # Calculate cross-validation statistics
    mean_f1 = np.mean(cv_scores['f1'])
    std_f1 = np.std(cv_scores['f1'])
    mean_auc = np.mean(cv_scores['auc'])
    std_auc = np.std(cv_scores['auc'])
    mean_acc = np.mean(cv_scores['accuracy'])
    std_acc = np.std(cv_scores['accuracy'])

    print(f"\nCross-validation complete for {experiment_name}!")
    print(f"Mean F1 score: {mean_f1:.4f} (±{std_f1:.4f})")
    print(f"Mean AUC score: {mean_auc:.4f} (±{std_auc:.4f})")
    print(f"Mean Accuracy: {mean_acc:.4f} (±{std_acc:.4f})")
    print(f"Per-fold F1 scores: {cv_scores['f1']}")
    print(f"Per-fold AUC scores: {cv_scores['auc']}")
    print(f"Per-fold Accuracy scores: {cv_scores['accuracy']}")

    # Save cross-validation results
    cv_results = {
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

    with open(os.path.join(exp_dir, 'cross_validation_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=4)

    return cv_results


def plot_learning_curves(history, save_dir, experiment_name):
    """Plot and save learning curves for the experiment"""
    plt.figure(figsize=(15, 5))

    # Extract metrics
    epochs = range(1, len(history['train']) + 1)
    train_loss = [m['loss'] for m in history['train']]
    val_loss = [m['loss'] for m in history['val']]
    train_f1 = [m['f1'] for m in history['train']]
    val_f1 = [m['f1'] for m in history['val']]

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # F1 subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_f1, 'b-', label='Training F1')
    plt.plot(epochs, val_f1, 'r-', label='Validation F1')
    plt.title('F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle(f'Learning Curves - {experiment_name}')
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, f'learning_curves_{experiment_name}.png'), dpi=300)
    plt.close()


def create_summary_visualization(ablation_results, save_dir='ablation_results'):
    """Create summary visualizations comparing all ablation experiments"""
    # Extract metrics for comparison
    experiment_names = []
    f1_scores = []
    accuracies = []
    aucs = []

    # Extract standard deviations for error bars
    f1_stds = []
    acc_stds = []
    auc_stds = []

    for exp_name, result in ablation_results.items():
        experiment_names.append(exp_name)
        f1_scores.append(result['mean_f1'])
        accuracies.append(result['mean_accuracy'])
        aucs.append(result['mean_auc'])

        # Extract standard deviations
        f1_stds.append(result['std_f1'])
        acc_stds.append(result['std_accuracy'])
        auc_stds.append(result['std_auc'])

    # 1. Create bar charts for each metric with error bars
    # Set aesthetic parameters
    plt.figure(figsize=(15, 8))
    bar_width = 0.25
    index = np.arange(len(experiment_names))

    # Sort experiments by accuracy for consistent ordering
    sorted_indices = np.argsort(accuracies)[::-1]  # Descending order
    sorted_names = [experiment_names[i] for i in sorted_indices]

    sorted_accuracies = [accuracies[i] for i in sorted_indices]
    sorted_f1_scores = [f1_scores[i] for i in sorted_indices]
    sorted_aucs = [aucs[i] for i in sorted_indices]

    sorted_acc_stds = [acc_stds[i] for i in sorted_indices]
    sorted_f1_stds = [f1_stds[i] for i in sorted_indices]
    sorted_auc_stds = [auc_stds[i] for i in sorted_indices]

    # Create grouped bar chart
    plt.figure(figsize=(14, 8))
    bars1 = plt.bar(index, sorted_accuracies, bar_width, label='Accuracy', color='#3274A1',
                    yerr=sorted_acc_stds, capsize=5)
    bars2 = plt.bar(index + bar_width, sorted_f1_scores, bar_width, label='F1 Score', color='#E1812C',
                    yerr=sorted_f1_stds, capsize=5)
    bars3 = plt.bar(index + 2 * bar_width, sorted_aucs, bar_width, label='AUC', color='#3A923A',
                    yerr=sorted_auc_stds, capsize=5)

    plt.xlabel('Experiment', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Performance Comparison Across Ablation Experiments', fontsize=16)
    plt.xticks(index + bar_width, sorted_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    plt.savefig(os.path.join(save_dir, 'metrics_comparison_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Create radar chart for multi-metric visualization
    # Prepare the data
    labels = ['Accuracy', 'F1 Score', 'AUC']

    # Number of variables
    N = len(labels)

    # Create angle for each variable
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Set figure size
    plt.figure(figsize=(12, 10))

    # Create subplot with polar projection
    ax = plt.subplot(111, polar=True)

    # Set ticks and labels
    plt.xticks(angles[:-1], labels, fontsize=12)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
    plt.ylim(0, 1)

    # Plot each experiment
    colors = plt.cm.viridis(np.linspace(0, 1, len(experiment_names)))
    for i, name in enumerate(experiment_names):
        values = [
            ablation_results[name]['mean_accuracy'],
            ablation_results[name]['mean_f1'],
            ablation_results[name]['mean_auc']
        ]
        values += values[:1]  # Close the loop

        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=name, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=10)

    plt.title('Ablation Study: Radar Plot of Metrics', fontsize=16, y=1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_radar_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--acc_dir', type=str,
                        default="D:/PyCharm/Project_PDFtransformer/University of Ottawa Electric Motor Dataset/acc")
    parser.add_argument('--sound_dir', type=str,
                        default="D:/PyCharm/Project_PDFtransformer/University of Ottawa Electric Motor Dataset/sound")
    parser.add_argument('--temp_dir', type=str,
                        default="D:/PyCharm/Project_PDFtransformer/University of Ottawa Electric Motor Dataset/temp")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_patches', type=int, default=48)
    parser.add_argument('--projection_dim', type=int, default=192)
    parser.add_argument('--save_dir', type=str, default="D:/PyCharm/Project_PDFtransformer/ablation_results2")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--lr_scheduler', type=str, default='cosine_warmup')
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mixup_alpha', type=float, default=0.3)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=2e-5)
    parser.add_argument('--use_image_input', action='store_true', help='Convert modality data to images')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=120, help='Number of epochs to train')
    parser.add_argument('--early_stopping', type=int, default=20, help='Early stopping patience')
    args = parser.parse_args()

    # Create saving directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Define the 8 required ablation experiments
    ablation_configs = [
        {
            'name': 'entire_mechanism',
            'description': 'Complete model with all components enabled'
            # All features enabled by default
        },
        {
            'name': 'no_PredictionModule',
            'description': 'Model with equal weights for all modalities',
            'disable_prediction_module': True
        },
        {
            'name': 'no_CustomDataset',
            'description': 'Model without data augmentation',
            'disable_data_augmentation': True
        },
        {
            'name': 'no_cross_attention',
            'description': 'Model without cross-modal attention mechanism',
            'disable_cross_attention': True
        },
        {
            'name': 'baseline',
            'description': 'Basic transformer fusion without prediction module, data augmentation, or cross-attention',
            'disable_prediction_module': True,
            'disable_cross_attention': True,
            'disable_data_augmentation': True
        },
        {
            'name': 'only_acc',
            'description': 'Using only acceleration modality',
            'use_only_modality': 'acc'
        },
        {
            'name': 'only_sound',
            'description': 'Using only sound modality',
            'use_only_modality': 'sound'
        },
        {
            'name': 'only_temp',
            'description': 'Using only temperature modality',
            'use_only_modality': 'temp'
        }
    ]

    # Run each ablation experiment with cross-validation
    cv_results = {}
    for config in ablation_configs:
        result = run_ablation_experiment_with_cv(config, args, args.save_dir)
        cv_results[config['name']] = result

    # Save overall results
    with open(os.path.join(args.save_dir, 'all_cv_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=4)

    # Create summary visualizations
    create_summary_visualization(cv_results, args.save_dir)

    # Print final ranking by F1 score
    print("\n=== Ablation Study Results (Ranked by F1 Score) ===")
    experiments_by_f1 = sorted(cv_results.items(), key=lambda x: x[1]['mean_f1'], reverse=True)
    for i, (exp_name, exp_result) in enumerate(experiments_by_f1):
        print(f"{i + 1}. {exp_name}: F1={exp_result['mean_f1']:.4f}±{exp_result['std_f1']:.4f}, "
              f"Accuracy={exp_result['mean_accuracy']:.4f}±{exp_result['std_accuracy']:.4f}, "
              f"AUC={exp_result['mean_auc']:.4f}±{exp_result['std_auc']:.4f}")

    print("\nAblation study with cross-validation completed! Results saved to", args.save_dir)


if __name__ == "__main__":
    main()
