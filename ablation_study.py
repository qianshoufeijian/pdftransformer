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
from sklearn.model_selection import train_test_split
import copy

from PDFtransformer import TransformerWithPDF, CustomDataset, EnhancedMixedLoss, get_enhanced_lr_scheduler
from main import preprocess_data, train_with_mixed_precision, evaluate, set_seed, calculate_metrics


class AblationTransformerWithPDF(TransformerWithPDF):
    def __init__(self, input_shape, num_heads, num_patches, projection_dim, num_classes,
                 image_input=False, dropout_rate=0.3,
                 # Ablation parameters
                 disable_cross_attention=False,
                 disable_residual=False,
                 disable_dynamic_weights=False,
                 disable_multi_supervision=False,
                 fusion_method='transformer',  # 'transformer', 'concat', 'average', 'max', 'attention'
                 use_only_modality=None):  # None, 'acc', 'sound', 'temp'

        # Initialize parent class
        super(AblationTransformerWithPDF, self).__init__(
            input_shape, num_heads, num_patches, projection_dim, num_classes,
            image_input, dropout_rate
        )

        # Save ablation parameters
        self.disable_cross_attention = disable_cross_attention
        self.disable_residual = disable_residual
        self.disable_dynamic_weights = disable_dynamic_weights
        self.disable_multi_supervision = disable_multi_supervision
        self.fusion_method = fusion_method
        self.use_only_modality = use_only_modality

        # For simple fusion methods, replace the fusion_layer
        if fusion_method == 'concat':
            self.simple_fusion = nn.Sequential(
                nn.Linear(projection_dim * 3, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU()
            )
        elif fusion_method in ['average', 'max']:
            # No parameters needed
            pass
        elif fusion_method == 'attention':
            self.attention_fusion = nn.MultiheadAttention(
                embed_dim=projection_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(projection_dim)

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

        # Apply modality-specific transformers with optional residual connections
        for layer in self.acc_transformer:
            acc_patches = layer(acc_patches)
            if not self.disable_residual:
                acc_patches = acc_patches + 0.1 * acc_original

        for layer in self.sound_transformer:
            sound_patches = layer(sound_patches)
            if not self.disable_residual:
                sound_patches = sound_patches + 0.1 * sound_original

        for layer in self.temp_transformer:
            temp_patches = layer(temp_patches)
            if not self.disable_residual:
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
        if not self.disable_dynamic_weights:
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

        # Apply appropriate fusion method
        if self.fusion_method == 'transformer':
            # Use the original transformer fusion layer
            if self.disable_cross_attention:
                # When cross-attention is disabled, we modify the fusion layer behavior
                # by concatenating the features directly
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
        elif self.fusion_method == 'concat':
            # Simple concatenation followed by linear projection
            acc_mean = acc_patches.mean(dim=1)
            sound_mean = sound_patches.mean(dim=1)
            temp_mean = temp_patches.mean(dim=1)
            concat_features = torch.cat([acc_mean, sound_mean, temp_mean], dim=1)
            fused_feat = self.simple_fusion(concat_features)
            # Reshape to match expected output format of fusion_layer
            fused_features = fused_feat.unsqueeze(1).repeat(1, acc_patches.size(1), 1)
        elif self.fusion_method == 'average':
            # Simple averaging
            fused_features = (acc_patches + sound_patches + temp_patches) / 3
        elif self.fusion_method == 'max':
            # Element-wise max
            fused_features = torch.maximum(torch.maximum(acc_patches, sound_patches), temp_patches)
        elif self.fusion_method == 'attention':
            # Self-attention based fusion
            # Concatenate features, perform self-attention, then reduce
            concat_feats = torch.cat([acc_patches, sound_patches, temp_patches], dim=1)
            attn_output, _ = self.attention_fusion(concat_feats, concat_feats, concat_feats)
            fused_features = self.fusion_norm(attn_output[:, :acc_patches.size(1)])

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

        if self.disable_multi_supervision or not self.training:
            # If multi-supervision is disabled or during inference, just use main logits
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


def run_ablation_experiment(config, args, save_dir='ablation_results'):
    """Run a single ablation experiment with the given configuration"""
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

    # Use a subset of data for faster ablation studies if requested
    if args.quick_ablation:
        # Use only 30% of the data for quick testing
        train_acc, _, train_sound, _, train_temp, _, train_labels, _ = train_test_split(
            train_acc, train_sound, train_temp, train_labels, test_size=0.7, random_state=args.seed)

        val_acc, _, val_sound, _, val_temp, _, val_labels, _ = train_test_split(
            val_acc, val_sound, val_temp, val_labels, test_size=0.7, random_state=args.seed)

    # Create data loaders
    train_dataset = CustomDataset(
        train_acc, train_sound, train_temp, train_labels,
        transform=True,
        aug_strength=0.5,  # Fixed aug_strength for ablation
        convert_to_image=args.use_image_input,
        image_size=(args.image_size, args.image_size)
    )

    val_dataset = CustomDataset(
        val_acc, val_sound, val_temp, val_labels,
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
        input_shape=[train_acc.shape[1], train_sound.shape[1], train_temp.shape[1]],
        num_heads=args.num_heads,
        num_patches=args.num_patches,
        projection_dim=args.projection_dim,
        num_classes=args.num_classes,
        image_input=args.use_image_input,
        dropout_rate=args.dropout_rate,
        # Ablation parameters from config
        disable_cross_attention=config.get('disable_cross_attention', False),
        disable_residual=config.get('disable_residual', False),
        disable_dynamic_weights=config.get('disable_dynamic_weights', False),
        disable_multi_supervision=config.get('disable_multi_supervision', False),
        fusion_method=config.get('fusion_method', 'transformer'),
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
    history = {'train': [], 'val': []}

    for epoch in range(args.ablation_epochs):
        print(f"\nEpoch {epoch + 1}/{args.ablation_epochs}")

        # Train
        train_metrics = train_with_mixed_precision(
            model, train_loader, optimizer, criterion, device,
            epoch, args.ablation_epochs, scheduler, args.mixup_alpha, args.grad_clip
        )
        history['train'].append(train_metrics)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, args.num_classes)
        history['val'].append(val_metrics)

        print(f"Train Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}")

        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pth'))
            print(f"New best model saved with F1: {best_f1:.4f}")

        # Update scheduler if needed
        if scheduler is not None and args.lr_scheduler == 'plateau':
            scheduler.step(val_metrics['f1'])
        elif scheduler is not None and not hasattr(scheduler, 'step_batch'):
            scheduler.step()

    # Test final model performance
    model.load_state_dict(torch.load(os.path.join(exp_dir, 'best_model.pth')))
    final_metrics = evaluate(model, val_loader, criterion, device, args.num_classes)

    # Save results
    results = {
        'experiment': experiment_name,
        'configuration': config,
        'best_f1': best_f1,
        'final_metrics': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                          for k, v in final_metrics.items() if k != 'confusion_matrix'},
        'num_parameters': num_params,
        'args': vars(args)
    }

    # Save confusion matrix separately
    if 'confusion_matrix' in final_metrics:
        plt.figure(figsize=(10, 8))
        sns.heatmap(final_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f"C{i}" for i in range(args.num_classes)],
                    yticklabels=[f"C{i}" for i in range(args.num_classes)])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {experiment_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'confusion_matrix.png'))
        plt.close()

    # Save learning curves
    plot_learning_curves(history, exp_dir, experiment_name)

    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nExperiment {experiment_name} completed.")
    print(f"Best F1 Score: {best_f1:.4f}")

    return best_f1, final_metrics


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

    plt.savefig(os.path.join(save_dir, 'learning_curves.png'))
    plt.close()


def create_summary_visualization(ablation_results, save_path='ablation_summary.png'):
    """Create a summary visualization comparing all ablation experiments"""
    # Extract metrics for comparison
    experiment_names = []
    f1_scores = []
    accuracies = []
    precisions = []
    recalls = []

    for exp_name, result in ablation_results.items():
        experiment_names.append(exp_name)
        metrics = result['final_metrics']
        f1_scores.append(metrics['f1'])
        accuracies.append(metrics['accuracy'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])

    # Create figure with multiple comparisons
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Plot metrics
    metrics_data = [f1_scores, accuracies, precisions, recalls]
    metric_names = ['F1 Score', 'Accuracy', 'Precision', 'Recall']
    colors = ['#2C7BB6', '#D7191C', '#FF7F00', '#33A02C']

    for i, (ax, data, name, color) in enumerate(zip(axes.flatten(), metrics_data, metric_names, colors)):
        # Sort experiments by performance for this metric
        sorted_indices = np.argsort(data)
        sorted_names = [experiment_names[j] for j in sorted_indices]
        sorted_data = [data[j] for j in sorted_indices]

        # Plot horizontal bars
        bars = ax.barh(sorted_names, sorted_data, color=color, alpha=0.8)
        ax.set_title(name, fontsize=14)
        ax.set_xlim(min(0.5, min(data) * 0.95), max(1.0, max(data) * 1.05))
        ax.grid(True, axis='x', alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, sorted_data):
            ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f'{value:.4f}',
                    va='center', fontsize=9)

    plt.suptitle('Ablation Study Results', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create a radar chart for multi-metric visualization
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)

    # Number of metrics
    N = 4
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Plot each experiment
    colors = plt.cm.viridis(np.linspace(0, 1, len(experiment_names)))
    for i, exp_name in enumerate(experiment_names):
        values = [
            ablation_results[exp_name]['final_metrics']['f1'],
            ablation_results[exp_name]['final_metrics']['accuracy'],
            ablation_results[exp_name]['final_metrics']['precision'],
            ablation_results[exp_name]['final_metrics']['recall']
        ]
        values += values[:1]  # Close the loop

        ax.plot(angles, values, 'o-', linewidth=2, markersize=4, color=colors[i], label=exp_name)
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    # Set angle labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['F1', 'Accuracy', 'Precision', 'Recall'])

    # Add legend (place outside the plot for clarity)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)

    plt.title('Multi-metric Comparison', fontsize=15)
    plt.tight_layout()
    plt.savefig('ablation_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--acc_dir', type=str, default="D:/PyCharm/Project_PDFtransformer/University of Ottawa Electric Motor Dataset/acc")
    parser.add_argument('--sound_dir', type=str, default="D:/PyCharm/Project_PDFtransformer/University of Ottawa Electric Motor Dataset/sound")
    parser.add_argument('--temp_dir', type=str, default="D:/PyCharm/Project_PDFtransformer/University of Ottawa Electric Motor Dataset/temp")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--ablation_epochs', type=int, default=120)  # Fewer epochs for ablation
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_patches', type=int, default=48)
    parser.add_argument('--projection_dim', type=int, default=192)
    parser.add_argument('--save_dir', type=str, default="D:/PyCharm/Project_PDFtransformer/ablation_results")
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
    parser.add_argument('--quick_ablation', action='store_true', help='Use subset of data for faster ablation')
    parser.add_argument('--selected_experiments', type=str, default='all',
                        help='Comma-separated list of experiment names to run, or "all" to run all experiments')
    args = parser.parse_args()

    # Create saving directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Define all possible ablation experiments
    all_ablation_configs = [
        {
            'name': 'baseline',
            'description': 'Full model with all components enabled'
            # No ablations - all default settings
        },
        {
            'name': 'no_cross_attention',
            'description': 'Model without cross-attention between modalities',
            'disable_cross_attention': True
        },
        {
            'name': 'no_residual',
            'description': 'Model without residual connections',
            'disable_residual': True
        },
        {
            'name': 'no_dynamic_weights',
            'description': 'Model with equal weights for all modalities',
            'disable_dynamic_weights': True
        },
        {
            'name': 'no_multi_supervision',
            'description': 'Model with only the main classifier',
            'disable_multi_supervision': True
        },
        {
            'name': 'fusion_concat',
            'description': 'Simple concatenation fusion instead of transformer',
            'fusion_method': 'concat'
        },
        {
            'name': 'fusion_average',
            'description': 'Simple averaging fusion instead of transformer',
            'fusion_method': 'average'
        },
        {
            'name': 'fusion_max',
            'description': 'Simple max fusion instead of transformer',
            'fusion_method': 'max'
        },
        {
            'name': 'fusion_attention',
            'description': 'Simple attention-based fusion',
            'fusion_method': 'attention'
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
        },
        {
            'name': 'minimal',
            'description': 'Minimal model with most features disabled',
            'disable_cross_attention': True,
            'disable_residual': True,
            'disable_dynamic_weights': True,
            'disable_multi_supervision': True,
            'fusion_method': 'average'
        }
    ]

    # Filter experiments if specific ones are requested
    if args.selected_experiments.lower() != 'all':
        selected_names = [name.strip() for name in args.selected_experiments.split(',')]
        ablation_configs = [config for config in all_ablation_configs if config['name'] in selected_names]
        if not ablation_configs:
            print(f"No valid experiment names found in: {args.selected_experiments}")
            print(f"Available experiments: {[config['name'] for config in all_ablation_configs]}")
            return
        print(f"Running {len(ablation_configs)} selected experiments: {selected_names}")
    else:
        ablation_configs = all_ablation_configs
        print(f"Running all {len(ablation_configs)} ablation experiments")

    # Run each ablation experiment
    results = {}
    for config in ablation_configs:
        best_f1, final_metrics = run_ablation_experiment(config, args, args.save_dir)
        results[config['name']] = {
            'best_f1': best_f1,
            'final_metrics': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                              for k, v in final_metrics.items() if k != 'confusion_matrix'},
            'config': config
        }

    # Save overall results
    with open(os.path.join(args.save_dir, 'all_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Create summary visualizations
    create_summary_visualization(results, os.path.join(args.save_dir, 'ablation_summary.png'))

    # Print final ranking by F1 score
    print("\n=== Ablation Study Results (Ranked by F1 Score) ===")
    experiments_by_f1 = sorted(results.items(), key=lambda x: x[1]['best_f1'], reverse=True)
    for i, (exp_name, exp_result) in enumerate(experiments_by_f1):
        print(f"{i + 1}. {exp_name}: F1={exp_result['best_f1']:.4f}")

    print("\nAblation study completed! Results saved to", args.save_dir)


if __name__ == "__main__":
    main()