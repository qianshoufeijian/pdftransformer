import matplotlib.pyplot as plt
import os
import numpy as np


def plot_cross_validation_summary(cv_results, save_path):
    """
    Plot comprehensive summary of cross validation results

    Parameters:
        cv_results: Dictionary with cross-validation results
        save_path: Path to save the summary plots
    """
    # Create figure with multiple subplots for different visualizations
    fig = plt.figure(figsize=(18, 12))

    # 1. F1 and AUC scores per fold
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)

    # Extract fold scores
    fold_f1_scores = cv_results['fold_f1_scores']
    fold_auc_scores = cv_results['fold_auc_scores']
    folds = range(1, len(fold_f1_scores) + 1)

    # Plot F1 scores
    bars1 = ax1.bar([x - 0.2 for x in folds], fold_f1_scores, width=0.4, color='skyblue', alpha=0.8, label='F1 Score')

    # Plot AUC scores
    bars2 = ax1.bar([x + 0.2 for x in folds], fold_auc_scores, width=0.4, color='lightgreen', alpha=0.8,
                    label='AUC Score')

    # Add mean lines
    ax1.axhline(y=cv_results['mean_f1'], color='blue', linestyle='--',
                label=f"Mean F1: {cv_results['mean_f1']:.4f} (±{cv_results['std_f1']:.4f})")
    ax1.axhline(y=cv_results['mean_auc'], color='green', linestyle='--',
                label=f"Mean AUC: {cv_results['mean_auc']:.4f} (±{cv_results['std_auc']:.4f})")

    # Add value labels above bars
    for bars, scores in [(bars1, fold_f1_scores), (bars2, fold_auc_scores)]:
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{score:.4f}', ha='center', va='bottom', rotation=0, fontsize=9)

    ax1.set_title('Performance Metrics Across Folds', fontsize=14)
    ax1.set_xlabel('Fold', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_ylim(0, 1.15 * max(max(fold_f1_scores), max(fold_auc_scores)))
    ax1.set_xticks(folds)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # 2. Standard deviation visualization
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    metrics = ['F1', 'AUC']
    means = [cv_results['mean_f1'], cv_results['mean_auc']]
    stds = [cv_results['std_f1'], cv_results['std_auc']]

    # Create error bars
    ax2.bar(metrics, means, yerr=stds, capsize=10, color=['skyblue', 'lightgreen'], alpha=0.8)

    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax2.text(i, mean + std + 0.01, f'{mean:.4f}±{std:.4f}', ha='center', va='bottom')

    ax2.set_title('Mean Performance with Standard Deviation', fontsize=14)
    ax2.set_ylim(0, 1.15 * max(means))
    ax2.grid(True, linestyle='--', alpha=0.4)

    # 3. Boxplot of fold scores
    ax3 = plt.subplot2grid((2, 3), (1, 0))
    box_data = [fold_f1_scores, fold_auc_scores]
    box = ax3.boxplot(box_data, patch_artist=True, labels=metrics)

    # Color boxes
    colors = ['skyblue', 'lightgreen']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    ax3.set_title('Distribution of Scores', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.4)

    # 4. Spider/Radar plot for multi-metric visualization
    ax4 = plt.subplot2grid((2, 3), (1, 1), colspan=2, polar=True)

    # Define metrics to include in radar chart
    radar_metrics = ['F1', 'AUC', '1-StdDev']  # Lower stddev is better, so we use 1-StdDev
    fold_values = []

    # Calculate normalized values for each fold
    max_f1 = max(fold_f1_scores)
    max_auc = max(fold_auc_scores)
    max_std = max(cv_results['std_f1'], cv_results['std_auc'])

    for i in range(len(fold_f1_scores)):
        fold_values.append([
            fold_f1_scores[i] / max_f1 if max_f1 > 0 else 0,  # Normalized F1
            fold_auc_scores[i] / max_auc if max_auc > 0 else 0,  # Normalized AUC
            1.0 - ((fold_f1_scores[i] - cv_results['mean_f1']) ** 2 / max_std if max_std > 0 else 0)
            # Normalized stability
        ])

    # Calculate angles for radar chart
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Plot radar chart for each fold
    for i, values in enumerate(fold_values):
        values += values[:1]  # Close the loop
        ax4.plot(angles, values, 'o-', linewidth=2, markersize=4, alpha=0.7, label=f'Fold {i + 1}')
        ax4.fill(angles, values, alpha=0.05)

    # Add labels
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(radar_metrics)
    ax4.set_title('Multi-metric Fold Performance', fontsize=14)
    ax4.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "cross_validation_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Create a separate figure for score variance analysis
    plt.figure(figsize=(12, 6))

    # Calculate variance from mean for each fold
    f1_variance = [(score - cv_results['mean_f1']) ** 2 for score in fold_f1_scores]
    auc_variance = [(score - cv_results['mean_auc']) ** 2 for score in fold_auc_scores]

    plt.bar([x - 0.2 for x in folds], f1_variance, width=0.4, color='skyblue', alpha=0.8, label='F1 Variance')
    plt.bar([x + 0.2 for x in folds], auc_variance, width=0.4, color='lightgreen', alpha=0.8, label='AUC Variance')

    plt.title('Score Variance Analysis per Fold', fontsize=14)
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Variance from Mean', fontsize=12)
    plt.xticks(folds)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "variance_analysis.png"), dpi=300)
    plt.close()