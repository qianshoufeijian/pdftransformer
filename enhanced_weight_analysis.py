"""
增强的权重物理意义分析模块（修复版）
"""

import numpy as np
from scipy import signal, stats
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import platform


# ==================== 中文字体配置 ====================
def setup_chinese_font():
    system = platform.system()
    if system == 'Windows':
        font_list = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
    elif system == 'Darwin':
        font_list = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
    else:
        font_list = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback']

    for font_name in font_list:
        try:
            matplotlib.rcParams['font.sans-serif'] = [font_name] + matplotlib.rcParams['font.sans-serif']
            matplotlib.rcParams['axes.unicode_minus'] = False
            return True
        except:
            continue
    return False


CHINESE_FONT_AVAILABLE = setup_chinese_font()
sns.set(style="whitegrid")


# ==================== ACC模态：FFT分析 ====================
def compute_acc_fft_features(data):
    data = np.asarray(data).flatten()
    n = len(data)

    default = {'total_energy': 0.1, 'spectral_entropy': 3.0, 'peak_to_avg_ratio': 1.0, 'dominant_freq_ratio': 0.1}

    if n == 0:
        return default

    try:
        fft_vals = np.fft.rfft(data)
        fft_power = np.abs(fft_vals) ** 2
        total_power = np.sum(fft_power) + 1e-10
        power_normalized = fft_power / total_power

        # 避免 log(0)
        power_normalized = np.clip(power_normalized, 1e-10, 1.0)

        return {
            'total_energy': float(np.sum(fft_power)),
            'spectral_entropy': float(-np.sum(power_normalized * np.log(power_normalized))),
            'peak_to_avg_ratio': float(np.max(fft_power) / (np.mean(fft_power) + 1e-10)),
            'dominant_freq_ratio': float(np.max(fft_power) / total_power)
        }
    except:
        return default


# ==================== SOUND模态：MFCC + 声学特征 ====================
def compute_sound_features(data, frame_size=256, hop_size=128):
    data = np.asarray(data).flatten()
    n = len(data)

    default = {
        'mfcc_energy': 0.0, 'mfcc_std': 0.1, 'ste_mean': 0.1, 'ste_std': 0.01,
        'zcr_mean': 0.1, 'spectral_flux': 0.1, 'spectral_centroid': 0.25, 'spectral_rolloff': 0.4
    }

    if n == 0:
        return default

    # 安全地调整参数
    frame_size = max(1, min(frame_size, n))
    hop_size = max(1, min(hop_size, frame_size))

    features = {}

    try:
        # 1. 短时能量
        if n >= frame_size:
            n_frames = max(1, (n - frame_size) // hop_size + 1)
            ste = []
            for i in range(n_frames):
                start = i * hop_size
                end = min(start + frame_size, n)
                if end > start:
                    frame = data[start:end]
                    ste.append(np.sum(frame ** 2))
            if ste:
                features['ste_mean'] = float(np.mean(ste))
                features['ste_std'] = float(np.std(ste)) if len(ste) > 1 else 0.01
            else:
                features['ste_mean'] = float(np.sum(data ** 2) / n)
                features['ste_std'] = 0.01
        else:
            features['ste_mean'] = float(np.sum(data ** 2) / n)
            features['ste_std'] = 0.01

        # 2. 过零率
        if n > 1:
            zero_crossings = np.where(np.diff(np.signbit(data)))[0]
            features['zcr_mean'] = float(len(zero_crossings) / n)
        else:
            features['zcr_mean'] = 0.1

        # 3. 频谱通量
        features['spectral_flux'] = 0.1  # 默认值
        if n >= frame_size * 2:
            n_frames = max(1, (n - frame_size) // hop_size + 1)
            flux_values = []
            prev_spectrum = None
            for i in range(n_frames):
                start = i * hop_size
                end = min(start + frame_size, n)
                if end > start:
                    frame = data[start:end]
                    spectrum = np.abs(np.fft.rfft(frame))
                    if prev_spectrum is not None and len(spectrum) == len(prev_spectrum):
                        flux_values.append(np.sum((spectrum - prev_spectrum) ** 2))
                    prev_spectrum = spectrum
            if flux_values:
                features['spectral_flux'] = float(np.mean(flux_values))

        # 4. 频谱质心和滚降点
        fft_vals = np.abs(np.fft.rfft(data))
        freqs = np.fft.rfftfreq(n)
        fft_sum = np.sum(fft_vals) + 1e-10
        features['spectral_centroid'] = float(np.sum(freqs * fft_vals) / fft_sum)

        cumsum = np.cumsum(fft_vals)
        if cumsum[-1] > 0:
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            features['spectral_rolloff'] = float(freqs[rolloff_idx[0]]) if len(rolloff_idx) > 0 else float(freqs[-1])
        else:
            features['spectral_rolloff'] = 0.4

        # 5. 简化MFCC
        n_fft = min(512, n)
        fft_power = np.abs(np.fft.rfft(data, n=n_fft)) ** 2
        n_filters = 13

        if len(fft_power) > 1:
            mel_filters = np.zeros((n_filters, len(fft_power)))
            for j in range(n_filters):
                center = int((j + 1) * len(fft_power) / (n_filters + 1))
                width = max(1, len(fft_power) // (n_filters + 1))
                start = max(0, center - width)
                end = min(len(fft_power), center + width)
                if end > start:
                    mel_filters[j, start:end] = 1.0 / (end - start)

            mel_energy = np.dot(mel_filters, fft_power)
            mel_energy = np.clip(mel_energy, 1e-10, None)
            log_mel_energy = np.log(mel_energy)
            features['mfcc_energy'] = float(np.mean(log_mel_energy))
            features['mfcc_std'] = float(np.std(log_mel_energy))
        else:
            features['mfcc_energy'] = 0.0
            features['mfcc_std'] = 0.1

        return features

    except Exception as e:
        return default


# ==================== TEMP模态：时域统计 ====================
def compute_temp_features(data):
    data = np.asarray(data).flatten()
    n = len(data)

    default = {
        'mean': 0.0, 'std': 0.1, 'range': 0.1, 'trend_slope': 0.0, 'trend_r_squared': 0.0,
        'change_rate_mean': 0.01, 'change_rate_max': 0.1, 'coefficient_of_variation': 0.1,
        'segment_trend_std': 0.01, 'skewness': 0.0, 'kurtosis': 0.0, 'percentile_range': 0.1,
        'autocorr': 0.5, 'entropy': 1.0
    }

    if n == 0:
        return default

    try:
        features = {}
        features['mean'] = float(np.mean(data))
        features['std'] = float(np.std(data)) if n > 1 else 0.1
        features['range'] = float(np.max(data) - np.min(data))

        # 趋势分析
        if n > 1:
            try:
                slope, _, r_value, _, _ = stats.linregress(np.arange(n), data)
                features['trend_slope'] = float(slope)
                features['trend_r_squared'] = float(r_value ** 2)
            except:
                features['trend_slope'] = 0.0
                features['trend_r_squared'] = 0.0
        else:
            features['trend_slope'] = 0.0
            features['trend_r_squared'] = 0.0

        # 变化率
        if n > 1:
            diff = np.diff(data)
            features['change_rate_mean'] = float(np.mean(np.abs(diff)))
            features['change_rate_max'] = float(np.max(np.abs(diff)))
        else:
            features['change_rate_mean'] = 0.01
            features['change_rate_max'] = 0.1

        features['coefficient_of_variation'] = float(features['std'] / (np.abs(features['mean']) + 1e-10))

        # 分段趋势
        n_segments = min(4, n // 10) if n >= 10 else 0
        if n_segments >= 2:
            segment_size = n // n_segments
            segment_means = []
            for i in range(n_segments):
                start = i * segment_size
                end = start + segment_size if i < n_segments - 1 else n
                segment_means.append(np.mean(data[start:end]))
            features['segment_trend_std'] = float(np.std(segment_means))
        else:
            features['segment_trend_std'] = 0.01

        # 偏度和峰度
        if n > 2:
            try:
                features['skewness'] = float(stats.skew(data))
                features['kurtosis'] = float(stats.kurtosis(data))
                # 处理可能的 nan
                if np.isnan(features['skewness']):
                    features['skewness'] = 0.0
                if np.isnan(features['kurtosis']):
                    features['kurtosis'] = 0.0
            except:
                features['skewness'] = 0.0
                features['kurtosis'] = 0.0
        else:
            features['skewness'] = 0.0
            features['kurtosis'] = 0.0

        # 百分位数范围
        if n > 4:
            features['percentile_range'] = float(np.percentile(data, 90) - np.percentile(data, 10))
        else:
            features['percentile_range'] = features['range']

        # 自相关
        if n > 2:
            try:
                autocorr = np.corrcoef(data[:-1], data[1:])[0, 1]
                features['autocorr'] = float(autocorr) if not np.isnan(autocorr) else 0.5
            except:
                features['autocorr'] = 0.5
        else:
            features['autocorr'] = 0.5

        # 信号熵
        if n > 1 and features['std'] > 1e-10:
            try:
                hist, _ = np.histogram(data, bins=min(10, max(2, n // 2)), density=True)
                hist = hist[hist > 0]
                if len(hist) > 0:
                    features['entropy'] = float(-np.sum(hist * np.log(hist + 1e-10)))
                else:
                    features['entropy'] = 1.0
            except:
                features['entropy'] = 1.0
        else:
            features['entropy'] = 1.0

        return features

    except Exception as e:
        return default


# ==================== 信号质量评分 ====================
def compute_acc_quality_score(features):
    energy_score = np.tanh(features['total_energy'] / 1000)
    entropy_score = np.clip(1 - np.abs(features['spectral_entropy'] - 3) / 3, 0, 1)
    peak_score = np.tanh(features['peak_to_avg_ratio'] / 10)
    quality = 0.4 * energy_score + 0.3 * entropy_score + 0.3 * peak_score
    return float(np.clip(quality, 0.01, 0.99))


def compute_sound_quality_score(features):
    ste_score = np.tanh(features['ste_mean'] / 100)
    zcr_score = np.clip(1 - np.abs(features['zcr_mean'] - 0.1) / 0.1, 0, 1)
    flux_score = np.tanh(features['spectral_flux'] / 1000)
    mfcc_score = np.clip(np.tanh((features['mfcc_energy'] + 10) / 10), 0, 1)
    quality = 0.3 * ste_score + 0.2 * zcr_score + 0.25 * flux_score + 0.25 * mfcc_score
    return float(np.clip(quality, 0.01, 0.99))


def compute_temp_quality_score(features):
    # 使用较大的缩放因子增加区分度
    change_score = np.tanh(features['change_rate_mean'] * 1000)
    trend_score = np.tanh(np.abs(features['trend_slope']) * 10000)
    cv_score = np.tanh(features['coefficient_of_variation'] * 100)
    range_score = np.tanh(features['range'] * 100)
    segment_score = np.tanh(features['segment_trend_std'] * 100)
    std_score = np.tanh(features['std'] * 100)
    skew_score = np.tanh(np.abs(features.get('skewness', 0)) * 2)
    kurt_score = np.tanh(np.abs(features.get('kurtosis', 0)))
    pct_score = np.tanh(features.get('percentile_range', 0) * 100)
    autocorr_score = 1 - np.abs(features.get('autocorr', 0.5))
    entropy_score = np.tanh(features.get('entropy', 1.0) * 2)

    quality = (0.12 * change_score +
               0.12 * trend_score +
               0.10 * cv_score +
               0.10 * range_score +
               0.10 * segment_score +
               0.12 * std_score +
               0.08 * skew_score +
               0.06 * kurt_score +
               0.08 * pct_score +
               0.06 * autocorr_score +
               0.06 * entropy_score)

    return float(np.clip(quality, 0.01, 0.99))


# ==================== 批量分析（修复版，不再使用固定的0.33）====================
def analyze_modality_enhanced(acc_data, sound_data, temp_data, labels=None):
    n_samples = acc_data.shape[0]
    print(f"Analyzing {n_samples} samples with enhanced methods...")
    print("  ACC: FFT analysis")
    print("  SOUND: MFCC + acoustic features")
    print("  TEMP: Time-domain statistics")

    acc_quality = np.zeros(n_samples)
    sound_quality = np.zeros(n_samples)
    temp_quality = np.zeros(n_samples)

    for i in range(n_samples):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{n_samples}")

        # 直接调用，不使用 try-except（因为函数内部已经处理了异常）
        acc_features = compute_acc_fft_features(acc_data[i])
        acc_quality[i] = compute_acc_quality_score(acc_features)

        sound_features = compute_sound_features(sound_data[i])
        sound_quality[i] = compute_sound_quality_score(sound_features)

        temp_features = compute_temp_features(temp_data[i])
        temp_quality[i] = compute_temp_quality_score(temp_features)

    # 归一化为权重
    total_quality = acc_quality + sound_quality + temp_quality + 1e-10
    enhanced_weights = np.stack([
        acc_quality / total_quality,
        sound_quality / total_quality,
        temp_quality / total_quality
    ], axis=1)

    # 打印统计信息验证
    print(f"\nQuality scores statistics:")
    print(f"  ACC   - mean: {np.mean(acc_quality):.4f}, std: {np.std(acc_quality):.4f}, "
          f"range: [{np.min(acc_quality):.4f}, {np.max(acc_quality):.4f}]")
    print(f"  SOUND - mean: {np.mean(sound_quality):.4f}, std: {np.std(sound_quality):.4f}, "
          f"range: [{np.min(sound_quality):.4f}, {np.max(sound_quality):.4f}]")
    print(f"  TEMP  - mean: {np.mean(temp_quality):.4f}, std: {np.std(temp_quality):.4f}, "
          f"range: [{np.min(temp_quality):.4f}, {np.max(temp_quality):.4f}]")

    print(f"\nEnhanced weights statistics:")
    print(f"  ACC   - std: {np.std(enhanced_weights[:,0]):.4f}")
    print(f"  SOUND - std: {np.std(enhanced_weights[:,1]):.4f}")
    print(f"  TEMP  - std: {np.std(enhanced_weights[:,2]):.4f}")

    return {'enhanced_weights': enhanced_weights, 'labels': labels}


# ==================== 相关性分析 ====================
def correlate_with_model_weights(enhanced_weights, model_weights, save_dir=None):
    modalities = ['ACC', 'SOUND', 'TEMP']
    methods = ['FFT', 'MFCC+STE', 'Time-domain']
    correlations = {}

    print("\n" + "=" * 60)
    print("Enhanced Features vs Model Weights Correlation")
    print("=" * 60)

    for i, (mod, method) in enumerate(zip(modalities, methods)):
        enhanced_w = enhanced_weights[:, i]
        model_w = model_weights[:, i]

        std_e = np.std(enhanced_w)
        std_m = np.std(model_w)

        print(f"\n{mod} ({method}):")
        print(f"  Enhanced std: {std_e:.6f}, Model std: {std_m:.6f}")

        if std_e < 1e-10 or std_m < 1e-10:
            pr, sr = 0.0, 0.0
            pp, sp = 1.0, 1.0
            print(f"  WARNING: Constant input detected!")
        else:
            try:
                pr, pp = pearsonr(enhanced_w, model_w)
                sr, sp = spearmanr(enhanced_w, model_w)
                pr = 0.0 if np.isnan(pr) else float(pr)
                sr = 0.0 if np.isnan(sr) else float(sr)
            except:
                pr, sr = 0.0, 0.0
                pp, sp = 1.0, 1.0

        print(f"  Pearson:  r={pr:.4f}, p={pp:.4e}")
        print(f"  Spearman: r={sr:.4f}, p={sp:.4e}")

        correlations[mod.lower()] = {
            'method': method,
            'pearson_r': float(pr), 'pearson_p': float(pp),
            'spearman_r': float(sr), 'spearman_p': float(sp)
        }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        _plot_correlation_figures(enhanced_weights, model_weights, methods, save_dir)

    return correlations


def _plot_correlation_figures(enhanced_weights, model_weights, methods, save_dir):
    modalities = ['ACC', 'SOUND', 'TEMP']
    colors = ['blue', 'green', 'red']

    # 散点图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Enhanced Features vs Model Weights', fontsize=14)
    for i, (ax, mod, color, method) in enumerate(zip(axes, modalities, colors, methods)):
        ax.scatter(enhanced_weights[:, i], model_weights[:, i], alpha=0.5, c=color, s=20)
        max_val = max(enhanced_weights[:, i].max(), model_weights[:, i].max(), 0.01)
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)

        std_e = np.std(enhanced_weights[:, i])
        std_m = np.std(model_weights[:, i])
        if std_e > 1e-10 and std_m > 1e-10:
            r, _ = pearsonr(enhanced_weights[:, i], model_weights[:, i])
            r = 0 if np.isnan(r) else r
        else:
            r = 0
        ax.set_xlabel(f'{method} Weight')
        ax.set_ylabel('Model Weight')
        ax.set_title(f'{mod} ({method})\nr={r:.3f}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scatter_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 趋势图
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    fig.suptitle('Weight Trends Comparison', fontsize=14)
    indices = np.arange(len(enhanced_weights))
    for i, (ax, mod, color, method) in enumerate(zip(axes, modalities, colors, methods)):
        ax.plot(indices, enhanced_weights[:, i], label=f'{method} Weight', color=color, alpha=0.7)
        ax.plot(indices, model_weights[:, i], label='Model Weight', color=color, linestyle='--', alpha=0.7)
        ax.set_ylabel('Weight')
        ax.set_title(f'{mod} ({method})')
        ax.legend()
        ax.set_xlim(0, len(indices)-1)
        ax.set_ylim(0, 1)
    axes[-1].set_xlabel('Sample Index')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'trend_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 均值柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(3)
    width = 0.35
    enhanced_means = enhanced_weights.mean(axis=0)
    model_means = model_weights.mean(axis=0)
    bars1 = ax.bar(x - width/2, enhanced_means, width, label='Enhanced Features', color='steelblue')
    bars2 = ax.bar(x + width/2, model_means, width, label='Model Weights', color='coral')
    ax.set_ylabel('Mean Weight')
    ax.set_title('Mean Weight Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{m}\n({method})' for m, method in zip(modalities, methods)])
    ax.legend()
    ax.set_ylim(0, 0.7)
    for bars, vals in [(bars1, enhanced_means), (bars2, model_means)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mean_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Figures saved to: {save_dir}")


def plot_spectrum_comparison(acc_data, sound_data, temp_data, sample_idx, save_path):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    data_list = [('ACC', acc_data[sample_idx], 'blue'),
                 ('SOUND', sound_data[sample_idx], 'green'),
                 ('TEMP', temp_data[sample_idx], 'red')]

    for i, (name, data, color) in enumerate(data_list):
        data = np.asarray(data).flatten()
        n = len(data)

        axes[i, 0].plot(data, color=color, linewidth=0.8)
        axes[i, 0].set_title(f'{name} - Time Domain')
        axes[i, 0].set_xlabel('Sample Point')
        axes[i, 0].set_ylabel('Amplitude')

        if n > 0:
            fft_mag = np.abs(np.fft.rfft(data))
            freqs = np.fft.rfftfreq(n)
            axes[i, 1].plot(freqs, fft_mag, color=color, linewidth=0.8)
            axes[i, 1].set_xlim(0, 0.5)
        axes[i, 1].set_title(f'{name} - FFT Spectrum')
        axes[i, 1].set_xlabel('Normalized Frequency')
        axes[i, 1].set_ylabel('Magnitude')

    plt.suptitle(f'Sample #{sample_idx} Time-Frequency Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_correlation_evolution(history, save_path, title='Correlation Evolution'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14)
    colors = {'acc': 'blue', 'sound': 'green', 'temp': 'red'}

    for ax, corr_type in zip(axes, ['pearson', 'spearman']):
        for mod in ['acc', 'sound', 'temp']:
            values = history.get(f'{mod}_{corr_type}', [])
            values = [0 if (v is None or np.isnan(v)) else v for v in values]
            if values:
                ax.plot(history['epochs'], values, color=colors[mod],
                        marker='o', markersize=4, label=mod.upper(), linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'{corr_type.capitalize()} Correlation')
        ax.set_title(f'{corr_type.capitalize()} Correlation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()