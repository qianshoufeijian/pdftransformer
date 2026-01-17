"""
增强版消融实验统计分析模块
Enhanced Ablation Study Statistical Analysis Module

功能：
1. 计算置信区间
2. 执行ANOVA和配对t检验
3. 量化各组件贡献度
4. 生成统计显著性热力图
5. 创建综合分析报告

作者：自动生成
日期：2026-01-05
"""

import numpy as np
import scipy.stats as stats
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import os
import argparse
from pathlib import Path


# ============================================================================
# 第一部分：统计计算函数
# ============================================================================

def compute_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    计算置信区间

    原理：
    - 使用t分布（适用于小样本，如5折交叉验证）
    - CI = mean ± t_{α/2, n-1} × SEM

    参数：
        data: 数值数组（如5折的分数）
        confidence: 置信水平（默认0.95表示95%置信区间）

    返回：
        (下界, 上界) 元组
    """
    n = len(data)
    if n < 2:
        return (float(data[0]), float(data[0])) if n == 1 else (0.0, 0.0)

    mean = np.mean(data)
    std_err = stats.sem(data)  # 标准误差 = std / √n

    # 使用t分布计算临界值
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_value * std_err

    return (float(mean - margin), float(mean + margin))


def perform_anova_test(results: Dict[str, Dict], metric: str = 'f1') -> Dict:
    """
    执行单因素方差分析（One-way ANOVA）

    原理：
    - H₀: 所有实验组的均值相等
    - H₁: 至少有两组均值不相等
    - F = 组间方差 / 组内方差

    参数：
        results: 消融实验结果字典
        metric:  要检验的指标 ('f1', 'accuracy', 'auc')

    返回：
        包含F统计量、p值等的字典
    """
    # 收集各实验的折叠分数
    groups = []
    group_names = []

    for exp_name, exp_result in results.items():
        key = f'fold_{metric}_scores'
        if key in exp_result:
            fold_scores = exp_result[key]
            groups.append(fold_scores)
            group_names.append(exp_name)

    if len(groups) < 2:
        return {
            'f_statistic': 0.0,
            'p_value': 1.0,
            'significant': False,
            'metric': metric,
            'error': 'Not enough groups for ANOVA'
        }

    # 执行单因素ANOVA
    f_stat, p_value = stats.f_oneway(*groups)

    return {
        'f_statistic': float(f_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'metric': metric,
        'num_groups': len(groups),
        'group_names': group_names
    }


def perform_paired_t_tests(results: Dict[str, Dict],
                           baseline_name: str = 'entire_mechanism',
                           metric: str = 'f1') -> Dict[str, Dict]:
    """
    执行配对t检验（与基准模型比较）

    原理：
    - 因为同一折使用相同的数据划分，观测值是配对的
    - H₀: 两组均值差为0
    - 计算Cohen's d效应量来衡量实际差异大小

    参数：
        results: 消融实验结果字典
        baseline_name: 基准实验名称
        metric:  要检验的指标

    返回：
        各实验与基准比较的t检验结果字典
    """
    key = f'fold_{metric}_scores'

    if baseline_name not in results or key not in results[baseline_name]:
        return {'error': f'Baseline {baseline_name} not found or missing {key}'}

    baseline_scores = np.array(results[baseline_name][key])
    t_test_results = {}

    for exp_name, exp_result in results.items():
        if exp_name == baseline_name:
            continue

        if key not in exp_result:
            continue

        exp_scores = np.array(exp_result[key])

        # 确保样本数相同
        if len(baseline_scores) != len(exp_scores):
            continue

        # 配对t检验（双尾）
        t_stat, p_value = stats.ttest_rel(baseline_scores, exp_scores)

        # 计算效应量（Cohen's d，配对样本版本）
        diff = baseline_scores - exp_scores
        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0

        t_test_results[exp_name] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'cohens_d': float(cohens_d),
            'mean_difference': float(np.mean(diff)),
            'std_difference': float(np.std(diff, ddof=1)),
            'effect_interpretation': interpret_cohens_d(cohens_d),
            'baseline_mean': float(np.mean(baseline_scores)),
            'experiment_mean': float(np.mean(exp_scores))
        }

    return t_test_results


def interpret_cohens_d(d: float) -> str:
    """
    解释Cohen's d效应量大小

    标准：
    - |d| < 0.2: 可忽略
    - 0.2 ≤ |d| < 0.5: 小效应
    - 0.5 ≤ |d| < 0.8: 中等效应
    - |d| ≥ 0.8: 大效应
    """
    d = abs(d)
    if d < 0.2:
        return 'negligible'
    elif d < 0.5:
        return 'small'
    elif d < 0.8:
        return 'medium'
    else:
        return 'large'


def compute_component_contribution(results: Dict[str, Dict], metric: str = 'f1') -> Dict[str, Dict]:
    """
    计算各组件的贡献度

    公式：
    - 贡献度 = 完整模型分数 - 移除该组件后的分数
    - 贡献百分比 = (贡献度 / 完整模型分数) × 100%

    参数：
        results: 消融实验结果字典
        metric: 要计算的指标

    返回：
        各组件贡献度字典
    """
    mean_key = f'mean_{metric}'

    if 'entire_mechanism' not in results:
        return {'error': 'entire_mechanism experiment not found'}

    full_model_score = results['entire_mechanism'].get(mean_key, 0)

    if full_model_score == 0:
        return {'error': f'Full model {mean_key} is 0 or not found'}

    contributions = {}

    # 模块贡献度映射
    component_mapping = {
        'PredictionModule（动态权重预测）': 'no_PredictionModule',
        'CustomDataset（数据增强）': 'no_CustomDataset',
        'Cross-Attention（跨模态注意力）': 'no_cross_attention',
    }

    for component_name, ablated_exp in component_mapping.items():
        if ablated_exp in results and mean_key in results[ablated_exp]:
            ablated_score = results[ablated_exp][mean_key]
            contribution = full_model_score - ablated_score
            contributions[component_name] = {
                'contribution': float(contribution),
                'percentage': float(contribution / full_model_score * 100),
                'full_model_score': float(full_model_score),
                'ablated_score': float(ablated_score)
            }

    # 多模态融合贡献（与最佳单模态比较）
    single_modal_exps = ['only_acc', 'only_sound', 'only_temp']
    best_single_modal_score = 0
    best_single_modal_name = None

    for exp_name in single_modal_exps:
        if exp_name in results and mean_key in results[exp_name]:
            score = results[exp_name][mean_key]
            if score > best_single_modal_score:
                best_single_modal_score = score
                best_single_modal_name = exp_name

    if best_single_modal_name:
        multimodal_contribution = full_model_score - best_single_modal_score
        contributions['多模态融合（vs最佳单模态）'] = {
            'contribution': float(multimodal_contribution),
            'percentage': float(multimodal_contribution / full_model_score * 100),
            'full_model_score': float(full_model_score),
            'best_single_modal': best_single_modal_name,
            'best_single_modal_score': float(best_single_modal_score)
        }

    # 各单模态的性能
    for exp_name in single_modal_exps:
        if exp_name in results and mean_key in results[exp_name]:
            score = results[exp_name][mean_key]
            contribution = full_model_score - score
            modality_name = exp_name.replace('only_', '').upper() + '模态'
            contributions[f'{modality_name}（单独使用）'] = {
                'contribution': float(contribution),
                'percentage': float(contribution / full_model_score * 100),
                'full_model_score': float(full_model_score),
                'single_modal_score': float(score)
            }

    return contributions


# ============================================================================
# 第二部分：可视化函数
# ============================================================================

def create_enhanced_results_table(results: Dict[str, Dict], save_dir: str) -> pd.DataFrame:
    """
    创建包含详细统计信息的结果表格
    """
    table_data = []

    for exp_name, exp_result in results.items():
        # 获取各折分数
        f1_scores = exp_result.get('fold_f1_scores', [])
        acc_scores = exp_result.get('fold_accuracy_scores', [])
        auc_scores = exp_result.get('fold_auc_scores', [])

        # 计算置信区间
        f1_ci = compute_confidence_interval(f1_scores) if f1_scores else (0, 0)
        acc_ci = compute_confidence_interval(acc_scores) if acc_scores else (0, 0)
        auc_ci = compute_confidence_interval(auc_scores) if auc_scores else (0, 0)

        row = {
            '实验名称': exp_name,
            '准确率': f"{exp_result.get('mean_accuracy', 0):.4f}",
            '准确率标准差': f"±{exp_result.get('std_accuracy', 0):.4f}",
            '准确率95%CI': f"[{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]",
            'F1分数': f"{exp_result.get('mean_f1', 0):.4f}",
            'F1标准差': f"±{exp_result.get('std_f1', 0):.4f}",
            'F1_95%CI': f"[{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]",
            'AUC': f"{exp_result.get('mean_auc', 0):.4f}",
            'AUC标准差': f"±{exp_result.get('std_auc', 0):.4f}",
            'AUC_95%CI': f"[{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]",
            '各折F1': str([f"{s:.4f}" for s in f1_scores])
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)

    # 按F1分数降序排序
    df = df.sort_values('F1分数', ascending=False)

    # 保存为CSV
    csv_path = os.path.join(save_dir, 'detailed_results_table.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"详细结果表格已保存至:  {csv_path}")

    # 保存为LaTeX格式（用于论文）
    latex_df = df[['实验名称', '准确率', '准确率标准差', 'F1分数', 'F1标准差', 'AUC', 'AUC标准差']]
    latex_path = os.path.join(save_dir, 'results_table.tex')
    latex_table = latex_df.to_latex(index=False, escape=False)
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    print(f"LaTeX表格已保存至:  {latex_path}")

    return df


def plot_component_contributions(contributions: Dict, save_dir: str, metric: str = 'F1'):
    """
    绘制组件贡献度柱状图
    """
    if 'error' in contributions:
        print(f"无法绘制贡献度图表:  {contributions['error']}")
        return

    # 分离模块贡献和模态贡献
    module_contribs = {}
    modality_contribs = {}

    for name, data in contributions.items():
        if '模态' in name or '多模态' in name:
            modality_contribs[name] = data
        else:
            module_contribs[name] = data

    # 创建双子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ===== 图1：模块贡献度 =====
    ax1 = axes[0]
    if module_contribs:
        names = list(module_contribs.keys())
        values = [module_contribs[n]['contribution'] for n in names]
        percentages = [module_contribs[n]['percentage'] for n in names]

        # 根据贡献正负设置颜色
        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
        bars = ax1.bar(range(len(names)), values, color=colors, edgecolor='black', linewidth=1.5)

        # 添加百分比标签
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            height = bar.get_height()
            offset = 0.002 if height >= 0 else -0.008
            ax1.annotate(f'{pct:.1f}%',
                         xy=(bar.get_x() + bar.get_width() / 2, height + offset),
                         ha='center', va='bottom' if height >= 0 else 'top',
                         fontsize=11, fontweight='bold')

        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=15, ha='right', fontsize=10)
        ax1.set_ylabel(f'{metric}分数贡献', fontsize=12)
        ax1.set_title('各模块对模型性能的贡献', fontsize=14, fontweight='bold')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
    else:
        ax1.text(0.5, 0.5, '无模块贡献数据', ha='center', va='center', fontsize=14)
        ax1.set_title('各模块对模型性能的贡献', fontsize=14, fontweight='bold')

    # ===== 图2：模态贡献度 =====
    ax2 = axes[1]
    if modality_contribs:
        names = list(modality_contribs.keys())
        # 简化名称显示
        short_names = [n.replace('（单独使用）', '').replace('（vs最佳单模态）', '') for n in names]
        values = [modality_contribs[n]['contribution'] for n in names]
        percentages = [modality_contribs[n]['percentage'] for n in names]

        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        bars = ax2.bar(range(len(names)), values, color=colors, edgecolor='black', linewidth=1.5)

        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            height = bar.get_height()
            offset = 0.002 if height >= 0 else -0.008
            ax2.annotate(f'{pct:.1f}%',
                         xy=(bar.get_x() + bar.get_width() / 2, height + offset),
                         ha='center', va='bottom' if height >= 0 else 'top',
                         fontsize=11, fontweight='bold')

        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(short_names, rotation=15, ha='right', fontsize=10)
        ax2.set_ylabel(f'{metric}分数贡献', fontsize=12)
        ax2.set_title('多模态融合贡献分析', fontsize=14, fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
    else:
        ax2.text(0.5, 0.5, '无模态贡献数据', ha='center', va='center', fontsize=14)
        ax2.set_title('多模态融合贡献分析', fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'component_contributions. png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"组件贡献度图表已保存至:  {save_path}")


def plot_fold_performance_boxplot(results: Dict[str, Dict], save_dir: str):
    """
    绘制各折性能分布箱线图
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    metrics = ['accuracy', 'f1', 'auc']
    titles = ['准确率 (Accuracy)', 'F1分数 (F1 Score)', 'AUC']

    for ax, metric, title in zip(axes, metrics, titles):
        data = []
        labels = []

        # 按均值排序
        sorted_exps = sorted(
            results.items(),
            key=lambda x: x[1].get(f'mean_{metric}', 0),
            reverse=True
        )

        for exp_name, exp_result in sorted_exps:
            key = f'fold_{metric}_scores'
            if key in exp_result:
                fold_scores = exp_result[key]
                data.append(fold_scores)
                # 简化实验名称
                short_name = exp_name.replace('_', '\n').replace('entire\nmechanism', 'Full\nModel')
                labels.append(short_name)

        if not data:
            ax.text(0.5, 0.5, f'无{title}数据', ha='center', va='center')
            continue

        # 绘制箱线图
        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        # 设置颜色
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # 添加散点（各折的实际值）
        for i, d in enumerate(data):
            x = np.random.normal(i + 1, 0.04, size=len(d))
            ax.scatter(x, d, alpha=0.7, s=60, c='red', edgecolors='black', linewidth=0.5, zorder=3)

        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title}分布（5折交叉验证）', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'fold_performance_boxplot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"各折性能箱线图已保存至: {save_path}")


def plot_statistical_significance_heatmap(results: Dict[str, Dict], save_dir: str, metric: str = 'f1'):
    """
    绘制统计显著性热力图（配对比较矩阵）
    """
    exp_names = list(results.keys())
    n_exp = len(exp_names)

    if n_exp < 2:
        print("实验数量不足，无法绘制显著性热力图")
        return

    # 创建p值矩阵
    p_matrix = np.ones((n_exp, n_exp))
    diff_matrix = np.zeros((n_exp, n_exp))

    key = f'fold_{metric}_scores'

    for i, exp1 in enumerate(exp_names):
        for j, exp2 in enumerate(exp_names):
            if i != j:
                if key in results[exp1] and key in results[exp2]:
                    scores1 = np.array(results[exp1][key])
                    scores2 = np.array(results[exp2][key])

                    if len(scores1) == len(scores2) and len(scores1) > 1:
                        _, p_value = stats.ttest_rel(scores1, scores2)
                        p_matrix[i, j] = p_value
                        diff_matrix[i, j] = np.mean(scores1) - np.mean(scores2)

    # 创建热力图
    fig, ax = plt.subplots(figsize=(12, 10))

    # 使用自定义颜色映射
    cmap = plt.cm.RdYlGn_r  # 红-黄-绿反转（小p值为红色）

    # 绘制热力图
    im = sns.heatmap(
        p_matrix,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        xticklabels=exp_names,
        yticklabels=exp_names,
        vmin=0,
        vmax=0.1,
        ax=ax,
        cbar_kws={'label': 'p-value'}
    )

    # 添加显著性标记
    for i in range(n_exp):
        for j in range(n_exp):
            if i != j:
                p = p_matrix[i, j]
                if p < 0.001:
                    marker = '***'
                elif p < 0.01:
                    marker = '**'
                elif p < 0.05:
                    marker = '*'
                else:
                    marker = ''

                if marker:
                    ax.text(j + 0.5, i + 0.75, marker,
                            ha='center', va='center',
                            fontsize=10, fontweight='bold', color='white')

    ax.set_title(f'配对t检验显著性热力图 ({metric.upper()})\n'
                 f'* p<0.05, ** p<0.01, *** p<0.001',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'significance_heatmap_{metric}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"显著性热力图已保存至:  {save_path}")


def plot_confidence_interval_chart(results: Dict[str, Dict], save_dir: str, metric: str = 'f1'):
    """
    绘制置信区间误差条图
    """
    exp_names = []
    means = []
    ci_lowers = []
    ci_uppers = []

    key_mean = f'mean_{metric}'
    key_fold = f'fold_{metric}_scores'

    # 按均值排序
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get(key_mean, 0),
        reverse=True
    )

    for exp_name, exp_result in sorted_results:
        if key_fold in exp_result:
            fold_scores = exp_result[key_fold]
            ci = compute_confidence_interval(fold_scores)
            mean = exp_result.get(key_mean, np.mean(fold_scores))

            exp_names.append(exp_name)
            means.append(mean)
            ci_lowers.append(mean - ci[0])
            ci_uppers.append(ci[1] - mean)

    if not exp_names:
        print(f"无{metric}数据，无法绘制置信区间图")
        return

    # 绘制
    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(exp_names))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(exp_names)))

    # 水平误差条图
    ax.barh(y_pos, means, xerr=[ci_lowers, ci_uppers],
            align='center', color=colors, edgecolor='black',
            capsize=5, error_kw={'linewidth': 2})

    # 添加数值标签
    for i, (mean, ci_l, ci_u) in enumerate(zip(means, ci_lowers, ci_uppers)):
        ax.text(mean + ci_u + 0.01, i, f'{mean:.4f}',
                va='center', ha='left', fontsize=10, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(exp_names, fontsize=11)
    ax.set_xlabel(f'{metric.upper()}分数', fontsize=12)
    ax.set_title(f'{metric.upper()}分数及95%置信区间', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'confidence_interval_{metric}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"置信区间图已保存至:  {save_path}")


# ============================================================================
# 第三部分：报告生成函数
# ============================================================================

def generate_comprehensive_report(results: Dict[str, Dict], save_dir: str) -> Dict:
    """
    生成完整的统计分析报告
    """
    report = {
        'summary': {},
        'anova_tests': {},
        'paired_t_tests': {},
        'component_contributions': {},
        'confidence_intervals': {}
    }

    # 1.  ANOVA检验
    print("\n执行ANOVA检验...")
    for metric in ['accuracy', 'f1', 'auc']:
        report['anova_tests'][metric] = perform_anova_test(results, metric)

    # 2. 配对t检验
    print("执行配对t检验...")
    for metric in ['accuracy', 'f1', 'auc']:
        report['paired_t_tests'][metric] = perform_paired_t_tests(results, 'entire_mechanism', metric)

    # 3. 组件贡献度
    print("计算组件贡献度...")
    for metric in ['accuracy', 'f1', 'auc']:
        report['component_contributions'][metric] = compute_component_contribution(results, metric)

    # 4. 置信区间
    print("计算置信区间...")
    for exp_name, exp_result in results.items():
        report['confidence_intervals'][exp_name] = {}
        for metric in ['accuracy', 'f1', 'auc']:
            key = f'fold_{metric}_scores'
            if key in exp_result:
                ci = compute_confidence_interval(exp_result[key])
                report['confidence_intervals'][exp_name][f'{metric}_95_ci'] = list(ci)

    # 保存JSON报告
    report_path = os.path.join(save_dir, 'statistical_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f"统计报告JSON已保存至:  {report_path}")

    # 生成文本摘要
    generate_text_summary(report, results, save_dir)

    return report


def generate_text_summary(report: Dict, results: Dict, save_dir: str):
    """
    生成人类可读的文本摘要报告
    """
    summary_path = os.path.join(save_dir, 'statistical_summary.txt')

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("                    消融实验统计分析报告\n")
        f.write("                 ABLATION STUDY STATISTICAL ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        # 1.  ANOVA结果
        f.write("1. 单因素方差分析 (One-way ANOVA)\n")
        f.write("-" * 50 + "\n")
        f.write("目的：检验各实验组之间是否存在显著差异\n\n")

        for metric, anova_result in report['anova_tests'].items():
            f.write(f"  【{metric.upper()}】\n")
            if 'error' not in anova_result:
                f.write(f"    F统计量:  {anova_result['f_statistic']:.4f}\n")
                f.write(f"    p值:  {anova_result['p_value']:.6f}\n")
                sig = "✓ 显著" if anova_result['significant'] else "✗ 不显著"
                f.write(f"    结论 (α=0.05): {sig}\n\n")
            else:
                f.write(f"    错误: {anova_result['error']}\n\n")

        # 2. 配对t检验结果
        f.write("\n2. 配对t检验 (与完整模型比较)\n")
        f.write("-" * 50 + "\n")
        f.write("目的：检验移除各组件后性能是否显著下降\n\n")

        for metric, t_tests in report['paired_t_tests'].items():
            f.write(f"  【{metric.upper()}】\n")
            if 'error' not in t_tests:
                for exp_name, t_result in t_tests.items():
                    sig_marker = "*" if t_result['significant'] else ""
                    if t_result['p_value'] < 0.001:
                        sig_marker = "***"
                    elif t_result['p_value'] < 0.01:
                        sig_marker = "**"

                    f.write(f"    {exp_name}:\n")
                    f.write(f"      均值差:  {t_result['mean_difference']:.4f}\n")
                    f.write(f"      p值: {t_result['p_value']:.4f}{sig_marker}\n")
                    f.write(f"      Cohen's d: {t_result['cohens_d']:. 3f} ({t_result['effect_interpretation']})\n")
            else:
                f.write(f"    错误: {t_tests['error']}\n")
            f.write("\n")

        # 3. 组件贡献度
        f.write("\n3. 组件贡献度分析\n")
        f.write("-" * 50 + "\n")

        f1_contribs = report['component_contributions'].get('f1', {})
        if 'error' not in f1_contribs:
            f.write("  【F1分数贡献】\n")
            for component, data in f1_contribs.items():
                f.write(f"    {component}:\n")
                f.write(f"      绝对贡献: {data['contribution']:.4f}\n")
                f.write(f"      相对贡献: {data['percentage']:.1f}%\n")
        f.write("\n")

        # 4. 各折详细结果
        f.write("\n4. 各折详细结果\n")
        f.write("-" * 50 + "\n\n")

        for exp_name, exp_result in results.items():
            f.write(f"  【{exp_name}】\n")

            for metric in ['f1', 'accuracy', 'auc']:
                key_fold = f'fold_{metric}_scores'
                key_mean = f'mean_{metric}'
                key_std = f'std_{metric}'

                if key_fold in exp_result:
                    fold_scores = exp_result[key_fold]
                    mean = exp_result.get(key_mean, np.mean(fold_scores))
                    std = exp_result.get(key_std, np.std(fold_scores))
                    ci = compute_confidence_interval(fold_scores)

                    f.write(f"    {metric.upper()}:\n")
                    f.write(f"      各折:  {[f'{s:.4f}' for s in fold_scores]}\n")
                    f.write(f"      均值±标准差: {mean:.4f} ± {std:. 4f}\n")
                    f.write(f"      95%置信区间:  [{ci[0]:.4f}, {ci[1]:. 4f}]\n")
            f.write("\n")

        # 5. 结论
        f.write("\n5. 主要结论\n")
        f.write("-" * 50 + "\n")

        # 找出贡献最大的组件
        if 'error' not in f1_contribs:
            sorted_contribs = sorted(
                [(k, v['percentage']) for k, v in f1_contribs.items() if isinstance(v, dict) and 'percentage' in v],
                key=lambda x: abs(x[1]),
                reverse=True
            )
            if sorted_contribs:
                f.write(f"  • 贡献最大的组件:  {sorted_contribs[0][0]} ({sorted_contribs[0][1]:.1f}%)\n")

        # 统计显著的组件
        t_tests_f1 = report['paired_t_tests'].get('f1', {})
        if 'error' not in t_tests_f1:
            sig_components = [name for name, data in t_tests_f1.items()
                              if isinstance(data, dict) and data.get('significant', False)]
            if sig_components:
                f.write(f"  • 统计显著的消融实验 (p<0.05): {', '.join(sig_components)}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("报告生成完成\n")

    print(f"文本摘要已保存至: {summary_path}")


# ============================================================================
# 第四部分：主函数和入口点
# ============================================================================

def run_enhanced_analysis(results_path: str, save_dir: str) -> Dict:
    """
    运行增强版统计分析的主函数

    参数：
        results_path: 消融实验结果JSON文件路径
        save_dir: 保存分析结果的目录

    返回：
        完整的统计分析报告字典
    """
    print("\n" + "=" * 60)
    print("      增强版消融实验统计分析")
    print("      Enhanced Ablation Study Statistical Analysis")
    print("=" * 60 + "\n")

    # 加载结果
    print(f"正在加载结果文件: {results_path}")
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    print(f"已加载 {len(results)} 个实验结果\n")

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 1. 生成详细结果表格
    print("=" * 40)
    print("步骤 1/6:  生成详细结果表格")
    print("=" * 40)
    create_enhanced_results_table(results, save_dir)

    # 2. 计算并可视化组件贡献度
    print("\n" + "=" * 40)
    print("步骤 2/6: 计算组件贡献度")
    print("=" * 40)
    contributions = compute_component_contribution(results, 'f1')
    plot_component_contributions(contributions, save_dir)

    # 3. 绘制各折性能箱线图
    print("\n" + "=" * 40)
    print("步骤 3/6: 绘制性能分布箱线图")
    print("=" * 40)
    plot_fold_performance_boxplot(results, save_dir)

    # 4. 绘制统计显著性热力图
    print("\n" + "=" * 40)
    print("步骤 4/6: 绘制统计显著性热力图")
    print("=" * 40)
    for metric in ['accuracy', 'f1', 'auc']:
        plot_statistical_significance_heatmap(results, save_dir, metric)

    # 5. 绘制置信区间图
    print("\n" + "=" * 40)
    print("步骤 5/6: 绘制置信区间图")
    print("=" * 40)
    for metric in ['accuracy', 'f1', 'auc']:
        plot_confidence_interval_chart(results, save_dir, metric)

    # 6. 生成综合报告
    print("\n" + "=" * 40)
    print("步骤 6/6: 生成综合统计报告")
    print("=" * 40)
    report = generate_comprehensive_report(results, save_dir)

    print("\n" + "=" * 60)
    print(f"分析完成！所有结果已保存至: {save_dir}")
    print("=" * 60)

    # 打印简要摘要
    print("\n>>> 快速摘要 <<<")
    for metric in ['f1', 'accuracy', 'auc']:
        anova = report['anova_tests'].get(metric, {})
        if 'error' not in anova:
            sig = "显著" if anova.get('significant', False) else "不显著"
            print(f"  {metric.upper()} ANOVA: F={anova.get('f_statistic', 0):.2f}, "
                  f"p={anova.get('p_value', 1):.4f} ({sig})")

    return report


def main():
    """
    命令行入口函数
    """
    parser = argparse.ArgumentParser(
        description='增强版消融实验统计分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法: 
  python enhanced_ablation_analysis.py --results_path ablation_results/all_cv_results.json --save_dir ablation_results/statistical_analysis

  或者在ablation_study. py完成后自动调用。
        """
    )

    parser.add_argument(
        '--results_path',
        type=str,
        default='ablation_results/all_cv_results.json',
        help='消融实验结果JSON文件路径 (默认: ablation_results/all_cv_results. json)'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        default='ablation_results/statistical_analysis',
        help='保存分析结果的目录 (默认: ablation_results/statistical_analysis)'
    )

    args = parser.parse_args()

    # 检查结果文件是否存在
    if not os.path.exists(args.results_path):
        print(f"错误: 结果文件不存在: {args.results_path}")
        print("请先运行 ablation_study.py 生成消融实验结果。")
        return

    # 运行分析
    report = run_enhanced_analysis(args.results_path, args.save_dir)

    return report


if __name__ == "__main__":
    main()