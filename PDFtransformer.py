import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import json
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter
import math
from scipy import signal
from PIL import Image
from torchvision import transforms
import cv2
import pywt
# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning)


# 添加新的组件
class ResidualBlock(nn.Module):
    """残差块 - 提高CNN特征提取能力"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差连接 - 如果维度变化，使用1x1卷积
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(identity)
        out = F.relu(out)

        return out


class SpatialAttention(nn.Module):
    """空间注意力模块 - 增强图像特征提取"""

    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # 计算通道维度上的平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 拼接特征
        x_cat = torch.cat([avg_out, max_out], dim=1)

        # 应用卷积和激活
        attention_map = torch.sigmoid(self.conv1(x_cat))

        # 应用注意力图
        return x * attention_map


class ChannelAttention(nn.Module):
    """通道注意力机制 - 关注重要特征通道"""

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享MLP
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # 平均池化分支
        avg_out = self.fc(self.avg_pool(x).view(b, c))

        # 最大池化分支
        max_out = self.fc(self.max_pool(x).view(b, c))

        # 合并两个分支
        attention = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)

        return x * attention

class ModalityToImage:
    """将时序模态数据转换为图像表示"""

    def __init__(self, image_size=(224, 224), methods=None, normalize=True):
        """
        初始化转换器

        参数:
            image_size: 输出图像的尺寸 (高度, 宽度)
            methods: 每个模态使用的转换方法字典 {'acc': 'method', 'sound': 'method', 'temp': 'method'}
                     可选方法: 'spectrogram', 'recurrence', 'gramian', 'scalogram', 'raw'
            normalize: 是否对生成的图像进行归一化
        """
        self.image_size = image_size
        self.normalize = normalize

        # 默认转换方法
        default_methods = {
            'acc': 'recurrence',  # 加速度数据适合递归图
            'sound': 'spectrogram',  # 声音数据适合频谱图
            'temp': 'gramian'  # 温度数据适合格拉姆矩阵
        }

        self.methods = methods if methods is not None else default_methods

        # 图像转换器
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) if normalize else transforms.Lambda(lambda x: x)
        ])

    def _create_spectrogram(self, data, n_fft=256, hop_length=32):
        """创建频谱图"""
        try:
            # 1. 数据准备
            data = np.asarray(data).flatten()  # 确保是numpy数组并展平

            # 2. 数据长度检查
            data_len = len(data)
            if data_len == 0:
                # 数据为空，返回空白图像
                blank_img = np.zeros((128, 128, 3), dtype=np.uint8)
                return Image.fromarray(blank_img)

            # 3. 自适应参数设置
            # 确保nperseg不超过数据长度
            nperseg = min(n_fft, data_len)

            # 确保hop_length合理
            if hop_length >= nperseg:
                # 如果hop_length太大，设为nperseg的四分之一
                hop_length = max(1, nperseg // 4)

            # 计算noverlap，并确保严格小于nperseg
            noverlap = nperseg - hop_length
            if noverlap >= nperseg:
                noverlap = nperseg - 1

            # 4. 计算短时傅里叶变换
            f, t, Zxx = signal.stft(data, nperseg=nperseg, noverlap=noverlap)

            # 5. 频谱图处理 - 取幅度的对数
            spectrogram = np.log1p(np.abs(Zxx))

            # 6. 归一化到[0,1]
            if np.max(spectrogram) > np.min(spectrogram):
                spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
            else:
                # 所有值相同的情况
                spectrogram = np.full_like(spectrogram, 0.5)

            # 7. 转换为RGB图像
            spectrogram_rgb = np.stack([spectrogram, spectrogram, spectrogram], axis=2)
            spectrogram_rgb = (spectrogram_rgb * 255).astype(np.uint8)

            return Image.fromarray(spectrogram_rgb)

        except Exception as e:
            # 捕获所有异常，返回空白图像
            blank_img = np.zeros((128, 128, 3), dtype=np.uint8)
            return Image.fromarray(blank_img)

    def _create_recurrence_plot(self, data, embed_dim=10, delay=2, threshold=None):
        """创建递归图"""
        # 确保数据是一维的
        data = data.flatten()

        # 如果数据太短，调整嵌入维度
        if len(data) < embed_dim * delay:
            embed_dim = max(2, len(data) // delay - 1)
            # 确保数据长度至少为 embed_dim * delay
        if len(data) < embed_dim * delay:
            # 填充数据到足够长度
            data = np.pad(data, (0, embed_dim * delay - len(data)), mode='constant')

        # 创建嵌入向量
        N = len(data) - (embed_dim - 1) * delay
        if N <= 0:  # 如果数据太短，简化处理
            embed_dim = 2
            delay = 1
            N = len(data) - (embed_dim - 1) * delay

        vectors = np.zeros((N, embed_dim))
        for i in range(N):
            for j in range(embed_dim):
                vectors[i, j] = data[i + j * delay]

        # 计算距离矩阵
        dist = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                d = np.linalg.norm(vectors[i] - vectors[j])
                dist[i, j] = d
                dist[j, i] = d

        # 应用阈值（如果指定）
        if threshold is not None:
            if threshold == 'auto':
                threshold = np.mean(dist) + np.std(dist)
            rp = (dist < threshold).astype(np.uint8) * 255
        else:
            # 归一化到[0,255]
            if np.max(dist) > np.min(dist):
                rp = 255 - ((dist - np.min(dist)) / (np.max(dist) - np.min(dist)) * 255).astype(np.uint8)
            else:
                rp = np.zeros_like(dist, dtype=np.uint8)

        # 转换为RGB图像
        rp_rgb = np.stack([rp, rp, rp], axis=2)

        return Image.fromarray(rp_rgb)

    def _create_gramian_angular_field(self, data):
        """创建格拉姆角场图像"""
        # 确保数据是一维的
        data = data.flatten()

        # 归一化到[-1,1]
        if np.max(data) > np.min(data):
            scaled_data = 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1
        else:
            scaled_data = np.zeros_like(data)

        # 转换到角度
        phi = np.arccos(scaled_data)

        # 计算格拉姆矩阵
        n = len(phi)
        gaf = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                gaf[i, j] = np.cos(phi[i] + phi[j])
                gaf[j, i] = gaf[i, j]  # 对称矩阵

        # 归一化到[0,1]
        gaf = (gaf + 1) / 2

        # 转换为RGB图像
        gaf_rgb = np.stack([gaf, gaf, gaf], axis=2)
        gaf_rgb = (gaf_rgb * 255).astype(np.uint8)

        return Image.fromarray(gaf_rgb)

    def _create_scalogram(self, data, scales=None):
        """创建小波变换图像"""
        # 确保数据是一维的
        data = data.flatten()

        # 设置默认尺度
        if scales is None:
            scales = np.arange(1, min(128, len(data) // 2))

        # 连续小波变换
        coef, freqs = signal.cwt(data, signal.morlet2, scales)

        # 取幅度的对数
        scalogram = np.log1p(np.abs(coef))

        # 归一化到[0,1]
        if np.max(scalogram) > np.min(scalogram):
            scalogram = (scalogram - np.min(scalogram)) / (np.max(scalogram) - np.min(scalogram))

        # 转换为RGB图像
        scalogram_rgb = np.stack([scalogram, scalogram, scalogram], axis=2)
        scalogram_rgb = (scalogram_rgb * 255).astype(np.uint8)

        return Image.fromarray(scalogram_rgb)

    def _create_raw_image(self, data, n_rows=None):
        """将原始数据重塑为2D图像"""
        # 确保数据是一维的
        data = data.flatten()

        # 确定行数和列数
        if n_rows is None:
            n_rows = int(np.sqrt(len(data)))

        n_cols = int(np.ceil(len(data) / n_rows))

        # 填充数据以匹配行列数
        padded_data = np.zeros(n_rows * n_cols)
        padded_data[:len(data)] = data

        # 重塑为2D
        img_data = padded_data.reshape(n_rows, n_cols)

        # 归一化到[0,1]
        if np.max(img_data) > np.min(img_data):
            img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

        # 转换为RGB图像
        img_rgb = np.stack([img_data, img_data, img_data], axis=2)
        img_rgb = (img_rgb * 255).astype(np.uint8)

        return Image.fromarray(img_rgb)

    def convert(self, data, modality):
        """
        将单个模态数据转换为图像

        参数:
            data: 输入数据
            modality: 模态类型 ('acc', 'sound', 'temp')

        返回:
            转换后的图像张量
        """
        method = self.methods.get(modality, 'raw')

        try:
            if method == 'spectrogram':
                img = self._create_spectrogram(data)
            elif method == 'recurrence':
                img = self._create_recurrence_plot(data)
            elif method == 'gramian':
                img = self._create_gramian_angular_field(data)
            elif method == 'scalogram':
                img = self._create_scalogram(data)
            else:  # 'raw'
                img = self._create_raw_image(data)

            # 应用转换
            return self.transform(img)

        except Exception as e:
            print(f"转换{modality}数据时出错: {e}")
            # 出错时返回空白图像
            blank = np.zeros((*self.image_size, 3), dtype=np.uint8)
            return self.transform(Image.fromarray(blank))


class CustomDataset(Dataset):
    def __init__(self, acc_data, sound_data, temp_data, labels, transform=False, scalers=None,
                 aug_strength=0.8, convert_to_image=False, image_size=(224, 224),
                 apply_denoise=True, filter_type="wavelet"):
        self.acc_data = acc_data
        self.sound_data = sound_data
        self.temp_data = temp_data
        self.labels = labels
        self.transform = transform
        self.aug_strength = aug_strength
        self.convert_to_image = convert_to_image
        self.apply_denoise = apply_denoise
        self.filter_type = filter_type  # 新增参数：滤波类型

        # 进行滤噪处理
        if self.apply_denoise:
            self.acc_data = self.apply_filters(self.acc_data, mode='acc')
            self.sound_data = self.apply_filters(self.sound_data, mode='sound')
            self.temp_data = self.apply_filters(self.temp_data, mode='temp')
        # 如果需要转换为图像，创建转换器
        if self.convert_to_image:
            # 为不同模态配置最适合的转换方法
            image_methods = {
                'acc': 'recurrence',  # 加速度数据适合递归图
                'sound': 'spectrogram',  # 声音数据适合频谱图
                'temp': 'gramian'  # 温度数据适合格拉姆矩阵
            }
            self.image_converter = ModalityToImage(
                image_size=image_size,
                methods=image_methods,
                normalize=True
            )

            # 跳过标准化
            self.scalers = None

            # 增强的图像转换 - 更强大的数据增强
            if self.transform:
                self.image_transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),  # 增大旋转角度
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),  # 更多颜色扰动
                    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),  # 添加缩放
                    transforms.RandomPerspective(distortion_scale=0.15, p=0.5),  # 增强透视变换
                    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.3),  # 添加高斯模糊
                    transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),  # 添加锐化
                ])
            else:
                self.image_transforms = None
        else:
            # 标准化处理
            if scalers is None:
                # 训练集上创建并拟合标准化器
                self.acc_scaler = StandardScaler()
                self.sound_scaler = StandardScaler()
                self.temp_scaler = StandardScaler()
                # Apply robust scaling with outlier handling
                acc_flat = self.acc_data.reshape(self.acc_data.shape[0], -1)
                sound_flat = self.sound_data.reshape(self.sound_data.shape[0], -1)
                temp_flat = self.temp_data.reshape(self.temp_data.shape[0], -1)

                # Clip extreme values (3 sigma rule)
                for data in [acc_flat, sound_flat, temp_flat]:
                    mean = np.mean(data, axis=0)
                    std = np.std(data, axis=0)
                    data_clipped = np.clip(data, mean - 3 * std, mean + 3 * std)
                self.acc_data = self.acc_scaler.fit_transform(self.acc_data)
                self.sound_data = self.sound_scaler.fit_transform(self.sound_data)
                self.temp_data = self.temp_scaler.fit_transform(self.temp_data)

                self.scalers = {
                    'acc': self.acc_scaler,
                    'sound': self.sound_scaler,
                    'temp': self.temp_scaler
                }
            else:
                # 验证集上使用训练集的标准化器
                self.acc_data = scalers['acc'].transform(self.acc_data)
                self.sound_data = scalers['sound'].transform(self.sound_data)
                self.temp_data = scalers['temp'].transform(self.temp_data)
                self.scalers = scalers

    def apply_filters(self, data, mode='acc'):
        filtered = data.copy()

        def apply_axis(func, x):
            return np.apply_along_axis(func, axis=1, arr=x)

        if self.filter_type in ['wavelet', 'all']:
            filtered = apply_axis(wavelet_denoise, filtered)

        if self.filter_type in ['lowpass', 'all']:
            filtered = apply_axis(lambda x: lowpass_filter(x, cutoff=0.1, fs=1.0), filtered)

        if self.filter_type in ['highpass', 'all']:
            filtered = apply_axis(lambda x: highpass_filter(x, cutoff=0.1, fs=1.0), filtered)

        return filtered

    def __len__(self):
        return len(self.labels)

    def _time_warp(self, x, sigma=0.2):
        """时间扭曲增强 - 比简单翻转更真实"""
        if x.shape[0] <= 3:  # 序列太短则跳过
            return x
        orig_steps = np.arange(x.shape[0])
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=x.shape[0])
        warp_steps = np.cumsum(random_warps)
        warp_steps = warp_steps / warp_steps[-1] * (x.shape[0] - 1)
        warped_x = np.interp(orig_steps, warp_steps, x)
        return warped_x

    def _freq_mask(self, x, num_masks=1, width_range=(0.05, 0.15)):
        """频域掩码 - 模拟传感器故障"""
        x_aug = x.copy()
        for i in range(num_masks):
            mask_width = int(x.shape[0] * np.random.uniform(*width_range))
            if mask_width > 0:
                mask_start = np.random.randint(0, x.shape[0] - mask_width)
                x_aug[mask_start:mask_start + mask_width] = 0
        return x_aug

    def _random_scaling(self, x, scale_range=(0.8, 1.2)):
        """应用随机缩放以模拟不同传感器灵敏度"""
        scale = np.random.uniform(*scale_range)
        return x * scale

    def _magnitude_warp(self, x, sigma=0.2):
        """幅度扭曲，创建随机平滑变化曲线"""
        if x.shape[0] <= 3:
            return x

        # 创建平滑曲线调整幅度
        knot_points = max(3, int(x.shape[0] / 10))
        knots = np.random.normal(loc=1.0, scale=sigma, size=knot_points)
        knot_positions = np.linspace(0, x.shape[0] - 1, knot_points)

        # 插值生成平滑曲线
        magnitude_changes = np.interp(np.arange(x.shape[0]), knot_positions, knots)

        return x * magnitude_changes

    def _jitter(self, x, sigma=0.05):
        """为每个点单独添加抖动噪声"""
        return x + np.random.normal(0, sigma, x.shape)

    def _window_slice(self, x, reduce_ratio=0.9):
        """提取信号的连续切片并插值回原始长度"""
        if x.shape[0] <= 3:
            return x

        new_len = int(x.shape[0] * reduce_ratio)
        if new_len <= 1:
            return x

        start_idx = np.random.randint(0, x.shape[0] - new_len + 1)
        sliced = x[start_idx:start_idx + new_len]

        # 重新调整到原始长度
        return np.interp(
            np.linspace(0, sliced.shape[0] - 1, x.shape[0]),
            np.arange(sliced.shape[0]),
            sliced
        )

    def _trend_injection(self, x):
        """为信号添加随机趋势"""
        trend_strength = np.random.uniform(0.1, 0.3) * self.aug_strength
        trend = np.linspace(0, 1, x.shape[0]) * trend_strength * np.random.choice([-1, 1])
        return x + trend

    def __getitem__(self, idx):
     # 获取原始数据
        acc = self.acc_data[idx].copy()
        sound = self.sound_data[idx].copy()
        temp = self.temp_data[idx].copy()
        label = self.labels[idx]


        if self.convert_to_image:
            # 转换为图像表示
            acc_img = self.image_converter.convert(acc, 'acc')
            sound_img = self.image_converter.convert(sound, 'sound')
            temp_img = self.image_converter.convert(temp, 'temp')

            # 应用图像增强（如果启用）
            if self.transform and self.image_transforms is not None:
                # 生成随机种子以保持一致性
                seed = torch.randint(0, 2 ** 32, (1,)).item()

                # 对每个模态应用相同的随机变换
                torch.manual_seed(seed)
                acc_img = self.image_transforms(acc_img)

                torch.manual_seed(seed)
                sound_img = self.image_transforms(sound_img)

                torch.manual_seed(seed)
                temp_img = self.image_transforms(temp_img)
                # 10%概率随机将一个模态替换为噪声，增强鲁棒性
                if np.random.random() < 0.1 * self.aug_strength:
                    noise_idx = np.random.randint(0, 3)
                    noise_img = torch.randn_like(acc_img) * 0.1  # 低强度噪声
                    if noise_idx == 0:
                        acc_img = (acc_img * 0.8) + noise_img
                    elif noise_idx == 1:
                        sound_img = (sound_img * 0.8) + noise_img
                    else:
                        temp_img = (temp_img * 0.8) + noise_img
            return acc_img, sound_img, temp_img, torch.tensor(label, dtype=torch.long)

        else:
            if self.transform:
                # 设置不同随机种子以避免模态间增强的相关性
                np.random.seed(int(idx * 1000 + np.random.randint(1000)))

                # 1. 高级噪声添加 - 不同噪声级别
                if np.random.random() < self.aug_strength * 0.7:
                    noise_level = np.random.uniform(0.01, 0.15 * self.aug_strength)
                    acc = acc + np.random.normal(0, noise_level, acc.shape)

                if np.random.random() < self.aug_strength * 0.7:
                    noise_level = np.random.uniform(0.01, 0.15 * self.aug_strength)
                    sound = sound + np.random.normal(0, noise_level, sound.shape)

                if np.random.random() < self.aug_strength * 0.5:
                    noise_level = np.random.uniform(0.005, 0.05 * self.aug_strength)
                    temp = temp + np.random.normal(0, noise_level, temp.shape)

                # 2. 时间扭曲 - 比简单翻转更真实
                if np.random.random() < self.aug_strength * 0.5:
                    acc = self._time_warp(acc, sigma=0.2 * self.aug_strength)

                if np.random.random() < self.aug_strength * 0.5:
                    sound = self._time_warp(sound, sigma=0.2 * self.aug_strength)

                if np.random.random() < self.aug_strength * 0.3:
                    temp = self._time_warp(temp, sigma=0.1 * self.aug_strength)

                # 3. 频域掩码 - 模拟传感器故障
                if np.random.random() < self.aug_strength * 0.6:
                    acc = self._freq_mask(acc, num_masks=int(1 + self.aug_strength * 2),
                                          width_range=(0.05, 0.2 * self.aug_strength))

                if np.random.random() < self.aug_strength * 0.6:
                    sound = self._freq_mask(sound, num_masks=int(1 + self.aug_strength * 2),
                                            width_range=(0.05, 0.2 * self.aug_strength))

                # 4. 随机缩放 - 模拟不同传感器灵敏度
                if np.random.random() < self.aug_strength * 0.5:
                    acc = self._random_scaling(acc,
                                               scale_range=(1.0 - 0.3 * self.aug_strength,
                                                            1.0 + 0.3 * self.aug_strength))

                if np.random.random() < self.aug_strength * 0.5:
                    sound = self._random_scaling(sound,
                                                 scale_range=(1.0 - 0.3 * self.aug_strength,
                                                              1.0 + 0.3 * self.aug_strength))

                # 5. 幅度扭曲 - 新增
                if np.random.random() < self.aug_strength * 0.5:
                    acc = self._magnitude_warp(acc, sigma=0.2 * self.aug_strength)

                if np.random.random() < self.aug_strength * 0.5:
                    sound = self._magnitude_warp(sound, sigma=0.2 * self.aug_strength)

                # 6. 抖动 - 新增
                if np.random.random() < self.aug_strength * 0.5:
                    acc = self._jitter(acc, sigma=0.05 * self.aug_strength)

                if np.random.random() < self.aug_strength * 0.5:
                    sound = self._jitter(sound, sigma=0.05 * self.aug_strength)

                # 7. 窗口切片 - 新增
                if np.random.random() < self.aug_strength * 0.4:
                    slice_ratio = np.random.uniform(0.85, 0.97)
                    acc = self._window_slice(acc, reduce_ratio=slice_ratio)
                    sound = self._window_slice(sound, reduce_ratio=slice_ratio)
                    temp = self._window_slice(temp, reduce_ratio=slice_ratio)

                # 8. 趋势注入 - 新增
                if np.random.random() < self.aug_strength * 0.3:
                    acc = self._trend_injection(acc)

                if np.random.random() < self.aug_strength * 0.3:
                    temp = self._trend_injection(temp)

                # 9. 翻转增强（较低概率）
                if np.random.random() < self.aug_strength * 0.3:
                    acc = acc[::-1].copy()
                    sound = sound[::-1].copy()
                    temp = temp[::-1].copy()

            return (
                torch.from_numpy(acc.copy()).float(),
                torch.from_numpy(sound.copy()).float(),
                torch.from_numpy(temp.copy()).float(),
                torch.tensor(label, dtype=torch.long)
            )


class PredictionModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PredictionModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        weights = F.softplus(self.fc3(x))  # 使用softplus确保权重为正且平滑
        return weights


class MultiModalFusion(nn.Module):
    def __init__(self, embed_dim, num_patches, num_heads, num_fusion_layers=4):  # 从3增加到4
        super(MultiModalFusion, self).__init__()
        self.embed_dim = embed_dim  # 保存为类属性
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)

        # 增加跨模态特征增强层
        self.modal_enhance = nn.ModuleDict({
            'acc': nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.LayerNorm(embed_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
            ),
            'sound': nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.LayerNorm(embed_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
            ),
            'temp': nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.LayerNorm(embed_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
            )
        })

        # 图像模态特有的注意力机制
        self.image_attention = nn.ModuleDict({
            'acc': nn.MultiheadAttention(embed_dim, num_heads // 2, dropout=0.1),
            'sound': nn.MultiheadAttention(embed_dim, num_heads // 2, dropout=0.1),
            'temp': nn.MultiheadAttention(embed_dim, num_heads // 2, dropout=0.1)
        })

        # 多层融合网络
        self.fusion_layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1),
                'norm1': nn.LayerNorm(embed_dim),
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(0.1)
                ),
                'norm2': nn.LayerNorm(embed_dim)
            }) for _ in range(num_fusion_layers)
        ])

        # 增强的模态间成对注意力机制 - 添加反向方向
        self.cross_modal_attention = nn.ModuleDict({
            'acc_sound': nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1),
            'acc_temp': nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1),
            'sound_temp': nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1),
            'sound_acc': nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1),  # 新增
            'temp_acc': nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1),  # 新增
            'temp_sound': nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)  # 新增
        })

        self.cross_norms = nn.ModuleDict({
            'acc': nn.LayerNorm(embed_dim),
            'sound': nn.LayerNorm(embed_dim),
            'temp': nn.LayerNorm(embed_dim)
        })

        # 添加门控融合机制
        self.fusion_gates = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 3),
            nn.Sigmoid()
        )

        self.modal_fc = nn.ModuleDict({
            'acc': nn.Linear(embed_dim, embed_dim),
            'sound': nn.Linear(embed_dim, embed_dim),
            'temp': nn.Linear(embed_dim, embed_dim)
        })
    def forward(self, acc_patches, sound_patches, temp_patches, acc_weights, sound_weights, temp_weights):
        batch_size = acc_patches.size(0)

        # 初始应用模态特定转换
        acc_patches = self.modal_fc['acc'](acc_patches)
        sound_patches = self.modal_fc['sound'](sound_patches)
        temp_patches = self.modal_fc['temp'](temp_patches)

        # 增强特征表示
        acc_enhanced = self.modal_enhance['acc'](acc_patches)
        sound_enhanced = self.modal_enhance['sound'](sound_patches)
        temp_enhanced = self.modal_enhance['temp'](temp_patches)

        # 应用跨补丁自注意力 - 专为图像特征设计
        acc_t = acc_enhanced.transpose(0, 1)
        sound_t = sound_enhanced.transpose(0, 1)
        temp_t = temp_enhanced.transpose(0, 1)

        acc_self, _ = self.image_attention['acc'](acc_t, acc_t, acc_t)
        sound_self, _ = self.image_attention['sound'](sound_t, sound_t, sound_t)
        temp_self, _ = self.image_attention['temp'](temp_t, temp_t, temp_t)

        # 原始和增强特征的残差连接
        acc_t = acc_t + 0.2 * acc_self
        sound_t = sound_t + 0.2 * sound_self
        temp_t = temp_t + 0.2 * temp_self

        # 添加CLS标记和位置编码
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        acc_input = torch.cat([cls_token, acc_patches], dim=1) + self.pos_embed
        sound_input = torch.cat([cls_token, sound_patches], dim=1) + self.pos_embed
        temp_input = torch.cat([cls_token, temp_patches], dim=1) + self.pos_embed

        # 应用动态权重，增加残差连接
        acc_input = acc_input * (1.0 + acc_weights.unsqueeze(1))
        sound_input = sound_input * (1.0 + sound_weights.unsqueeze(1))
        temp_input = temp_input * (1.0 + temp_weights.unsqueeze(1))

        # 转置为注意力层所需格式 (seq_len, batch, features)
        acc_t = acc_input.transpose(0, 1)
        sound_t = sound_input.transpose(0, 1)
        temp_t = temp_input.transpose(0, 1)

        # 双向跨模态注意力计算
        acc_sound, _ = self.cross_modal_attention['acc_sound'](acc_t, sound_t, sound_t)
        sound_acc, _ = self.cross_modal_attention['sound_acc'](sound_t, acc_t, acc_t)
        acc_temp, _ = self.cross_modal_attention['acc_temp'](acc_t, temp_t, temp_t)
        temp_acc, _ = self.cross_modal_attention['temp_acc'](temp_t, acc_t, acc_t)
        sound_temp, _ = self.cross_modal_attention['sound_temp'](sound_t, temp_t, temp_t)
        temp_sound, _ = self.cross_modal_attention['temp_sound'](temp_t, sound_t, sound_t)

        # 应用残差连接和层规范化，融合双向信息
        acc_t = self.cross_norms['acc'](acc_t + acc_sound + acc_temp)
        sound_t = self.cross_norms['sound'](sound_t + sound_acc + sound_temp)
        temp_t = self.cross_norms['temp'](temp_t + temp_acc + temp_sound)

        # 实现动态门控融合
        combined = torch.cat([
            acc_t.transpose(0, 1).mean(dim=1),
            sound_t.transpose(0, 1).mean(dim=1),
            temp_t.transpose(0, 1).mean(dim=1)
        ], dim=1)

        gate_outputs = self.fusion_gates(combined)  # (batch_size, 3 * embed_dim)
        gate_chunks = gate_outputs.chunk(3, dim=1)  # 3个(batch_size, embed_dim)形状的张量
        acc_gate, sound_gate, temp_gate = gate_chunks

        # 应用门控机制
        acc_gate = acc_gate.unsqueeze(0)  # (1, batch_size, embed_dim)
        sound_gate = sound_gate.unsqueeze(0)  # (1, batch_size, embed_dim)
        temp_gate = temp_gate.unsqueeze(0)  # (1, batch_size, embed_dim)
        acc_t = acc_t * acc_gate
        sound_t = sound_t * sound_gate
        temp_t = temp_t * temp_gate

        # 多层融合与残差连接
        fused = torch.cat([acc_t, sound_t, temp_t], dim=0)  # 沿序列维度拼接
        residual_fused = fused  # 用于残差连接

        for layer in self.fusion_layers:
            # 多头注意力
            attn_out, _ = layer['attn'](fused, fused, fused)
            fused = layer['norm1'](fused + attn_out)

            # 前馈网络
            ffn_out = layer['ffn'](fused)
            fused = layer['norm2'](fused + ffn_out)

            # 添加缩放残差连接防止梯度消失
            fused = fused + residual_fused * 0.1

        # 返回到batch-first格式
        return fused.transpose(0, 1)


class TransformerWithPDF(nn.Module):
    def __init__(self, input_shape, num_heads, num_patches, projection_dim, num_classes, image_input=False,
                 dropout_rate=0.3):
        super(TransformerWithPDF, self).__init__()
        self.input_shape = input_shape
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.image_input = image_input
        self.dropout_rate = dropout_rate  # Increased dropout for better regularization

        if image_input:
            # Create enhanced CNN encoders for image input
            self.acc_encoder = self._build_enhanced_cnn_encoder()
            self.sound_encoder = self._build_enhanced_cnn_encoder()
            self.temp_encoder = self._build_enhanced_cnn_encoder()
        else:
            # Enhanced encoders with proper normalizations
            self.acc_encoder = nn.Sequential(
                nn.Linear(2, self.projection_dim),  # 修正为 input_shape[0] -> projection_dim
                nn.LayerNorm(self.projection_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.projection_dim, self.projection_dim * num_patches),
                nn.Unflatten(1, (num_patches, self.projection_dim))
            )
            self.sound_encoder = self._build_similar_encoder(input_shape[1])
            self.temp_encoder = self._build_similar_encoder(input_shape[2])

        # Build transformer layers
        self.acc_transformer = nn.ModuleList([self._build_transformer_layer() for _ in range(2)])
        self.sound_transformer = nn.ModuleList([self._build_transformer_layer() for _ in range(2)])
        self.temp_transformer = nn.ModuleList([self._build_transformer_layer() for _ in range(2)])

        # Fusion and prediction modules
        self.fusion_layer = self._build_simplified_fusion()
        self.acc_prediction_module = PredictionModule(input_dim=self.projection_dim, output_dim=1)
        self.sound_prediction_module = PredictionModule(input_dim=self.projection_dim, output_dim=1)
        self.temp_prediction_module = PredictionModule(input_dim=self.projection_dim, output_dim=1)

        # Main transformer layers
        self.transformer_layers = nn.ModuleList([self._build_transformer_layer() for _ in range(6)])

        # Classifiers
        self.mid_classifier = self._build_enhanced_classifier(num_classes)
        self.acc_classifier = self._build_enhanced_classifier(num_classes)
        self.sound_classifier = self._build_enhanced_classifier(num_classes)
        self.temp_classifier = self._build_enhanced_classifier(num_classes)
        self.fc_out = self._build_enhanced_output_classifier(num_classes)

        self._initialize_weights()

    def _build_similar_encoder(self, input_dim):
        return nn.Sequential(
            nn.Linear(2, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.projection_dim, self.projection_dim * self.num_patches),
            nn.Unflatten(1, (self.num_patches, self.projection_dim))
        )

    def _build_enhanced_classifier(self, num_classes):
        return nn.Sequential(
            nn.Linear(self.projection_dim, self.projection_dim // 2),
            nn.LayerNorm(self.projection_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate * 0.7),  # Slightly lower dropout for classifiers
            nn.Linear(self.projection_dim // 2, num_classes)
        )

    def _build_enhanced_output_classifier(self, num_classes):
        return nn.Sequential(
            nn.Linear(self.projection_dim, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.projection_dim, self.projection_dim // 2),
            nn.LayerNorm(self.projection_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate * 0.7),
            nn.Linear(self.projection_dim // 2, num_classes)
        )

    def _build_simplified_fusion(self):
        # Simplified but still effective fusion layer
        return MultiModalFusion(
            embed_dim=self.projection_dim,
            num_patches=self.num_patches,
            num_heads=self.num_heads,
            num_fusion_layers=3  # Reduced from 4 to 3
        )

    def _initialize_weights(self):
        # Proper initialization for better convergence
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _build_enhanced_cnn_encoder(self):
        """构建增强版CNN编码器，处理图像输入"""
        return nn.Sequential(
            # 第一层 - 初始特征提取
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # [B, 64, 112, 112]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [B, 64, 56, 56]

            # 残差块1
            ResidualBlock(64, 128, stride=2),  # [B, 128, 28, 28]
            SpatialAttention(128),  # 添加空间注意力

            # 残差块2
            ResidualBlock(128, 256, stride=2),  # [B, 256, 14, 14]
            SpatialAttention(256),  # 添加空间注意力

            # 残差块3
            ResidualBlock(256, 512, stride=2),  # [B, 512, 7, 7]

            # 通道注意力
            ChannelAttention(512),

            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 512, 1, 1]
            nn.Flatten(),  # [B, 512]

            # 映射到所需维度
            nn.Linear(512, self.projection_dim * 2),
            nn.LayerNorm(self.projection_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.projection_dim * 2, self.projection_dim * self.num_patches),
            nn.Unflatten(1, (self.num_patches, self.projection_dim))  # [B, num_patches, projection_dim]
        )

    def _build_cnn_encoder(self):
        """构建CNN编码器，处理图像输入"""
        return self._build_enhanced_cnn_encoder()  # 使用增强版编码器

    def _build_transformer_layer(self):
        return nn.Sequential(
            nn.TransformerEncoderLayer(
                d_model=self.projection_dim,
                nhead=self.num_heads,
                dim_feedforward=self.projection_dim * 4,
                dropout=0.2,
                activation='gelu',
                batch_first=True
            ),
            nn.LayerNorm(self.projection_dim)
        )

    def forward(self, acc_data, sound_data, temp_data):

        # 编码每个模态
        if self.image_input:
            # 如果输入是图像，直接传递到CNN编码器
            acc_patches = self.acc_encoder(acc_data)
            sound_patches = self.sound_encoder(sound_data)
            temp_patches = self.temp_encoder(temp_data)
        else:
            # 原始时序数据编码
            acc_patches = self.acc_encoder(acc_data)
            sound_patches = self.sound_encoder(sound_data)
            temp_patches = self.temp_encoder(temp_data)

        # 保存原始patches用于残差连接
        acc_original = acc_patches
        sound_original = sound_patches
        temp_original = temp_patches

        # 应用模态特定transformer学习更好的表示，并添加残差连接
        for layer in self.acc_transformer:
            acc_patches = layer(acc_patches)
            acc_patches = acc_patches + 0.1 * acc_original  # 添加残差连接

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

        # 模态特定预测用于深度监督
        acc_pred = self.acc_classifier(acc_feat)
        sound_pred = self.sound_classifier(sound_feat)
        temp_pred = self.temp_classifier(temp_feat)

        # 模态动态权重计算
        acc_weights = self.acc_prediction_module(acc_feat)
        sound_weights = self.sound_prediction_module(sound_feat)
        temp_weights = self.temp_prediction_module(temp_feat)

        # 改进的权重归一化 - 使用softmax而不是简单除法
        weights = torch.cat([acc_weights, sound_weights, temp_weights], dim=1)
        normalized_weights = F.softmax(weights, dim=1)
        acc_weights = normalized_weights[:, 0].unsqueeze(1)
        sound_weights = normalized_weights[:, 1].unsqueeze(1)
        temp_weights = normalized_weights[:, 2].unsqueeze(1)

        # 改进的融合
        fused_features = self.fusion_layer(
            acc_patches, sound_patches, temp_patches,
            acc_weights, sound_weights, temp_weights
        )

        # 主Transformer处理和多点监督
        x = fused_features
        mid_features = None
        secondary_features = None

        for i, layer in enumerate(self.transformer_layers):
            x = layer(x)
            if i == len(self.transformer_layers) // 2:
                mid_features = x
            elif i == len(self.transformer_layers) // 4:
                secondary_features = x

        # 中间分类（如果可用）
        mid_logits = self.mid_classifier(mid_features[:, 0]) if mid_features is not None else \
            torch.zeros(x.size(0), self.fc_out[-1].out_features).to(x.device)

        # 从CLS标记进行主分类
        main_logits = self.fc_out(x[:, 0])
        if self.training:
            # 训练期间，使用所有预测并对辅助预测赋予更高权重
            final_logits = (main_logits * 0.6 +
                            mid_logits * 0.15 +
                            acc_pred * 0.1 +
                            sound_pred * 0.1 +
                            temp_pred * 0.05)
        else:
            # 推理期间，更关注主分类器
            final_logits = (main_logits * 0.7 +
                            mid_logits * 0.1 +
                            acc_pred * 0.08 +
                            sound_pred * 0.08 +
                            temp_pred * 0.04)
        return final_logits




# 修改EnhancedMixedLoss类
class EnhancedMixedLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1, temp=1.0, reduction='mean'):
        super().__init__()
        # 增加类别权重调整
        if alpha is not None:
            alpha = torch.sqrt(alpha)  # 使用平方根平滑权重
            alpha = alpha / alpha.sum() * len(alpha)
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.temp = temp
        self.reduction = reduction

    def forward(self, inputs, targets):
        targets = targets.long()

        # 增加困难样本检测
        with torch.no_grad():
            probs = F.softmax(inputs, dim=1)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze()
            sample_weights = 1.0 / (pt + 1e-5)  # 困难样本权重

        # 动态调整gamma
        dynamic_gamma = self.gamma * (1 + torch.sigmoid((pt - 0.5) * 10))

        # 改进的focal loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        focal_loss = (1 - pt) ** dynamic_gamma * ce_loss * sample_weights

        # 标签平滑
        if self.smoothing > 0:
            smooth_targets = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)
            smooth_targets = smooth_targets * (1 - self.smoothing) + self.smoothing / inputs.size(1)
            smooth_loss = -(smooth_targets * F.log_softmax(inputs, dim=1)).sum(dim=1)
            loss = 0.7 * focal_loss + 0.3 * smooth_loss
        else:
            loss = focal_loss

        return loss.mean()


def get_enhanced_lr_scheduler(optimizer, args, train_loader=None):
    if args.lr_scheduler == 'cosine_warm':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2)  # 更快的重启周期
    elif args.lr_scheduler == 'one_cycle':
        if train_loader is None:
            raise ValueError("train_loader must be provided for one_cycle scheduler")
        return optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr * 15,  # 更高的峰值学习率
            total_steps=args.epochs * len(train_loader),
            pct_start=0.3,  # 更快的热身
            div_factor=25.0,  # 更大的初始学习率除数
            final_div_factor=1000.0)  # 更小的最终学习率
    elif args.lr_scheduler == 'cosine_warmup':
        # 自定义热身+余弦退火调度器
        warmup_epochs = max(3, int(args.epochs * 0.1))

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif args.lr_scheduler == 'cyclic':
        # 更积极的循环学习率
        step_size_up = len(train_loader) * 2
        return optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=args.lr / 10, max_lr=args.lr * 10,
            step_size_up=step_size_up, mode='triangular2')
    elif args.lr_scheduler == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3,  # 减少耐心值
            threshold=0.001, verbose=True)
    return None


def get_lr_scheduler(optimizer, args, train_loader=None):
    if args.lr_scheduler == 'cosine_warm':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2)
    elif args.lr_scheduler == 'one_cycle':
        if train_loader is None:
            raise ValueError("train_loader must be provided for one_cycle scheduler")
        return optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr * 10,
            total_steps=args.epochs * len(train_loader))
    elif args.lr_scheduler == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    return None


# 修改balance_dataset函数
def balance_dataset(acc_data, sound_data, temp_data, labels, sampling_strategy='auto'):
    print(f"原始类别分布: {Counter(labels)}")

    # 改为按类别中位数进行过采样
    median_samples = np.median(np.bincount(labels))
    sampling_strategy = {cls: int(median_samples * 1.2) for cls in np.unique(labels)}

    # 使用SMOTE-NC处理混合类型数据
    smote = SMOTE(sampling_strategy=sampling_strategy,
                  k_neighbors=5,  # 减少邻居数量
                  random_state=42)

    # 合并特征时保留各模态特征
    combined_features = np.hstack([acc_data, sound_data, temp_data])
    resampled_features, resampled_labels = smote.fit_resample(combined_features, labels)

    # 分离不同模态的特征
    acc_dim = acc_data.shape[1]
    sound_dim = sound_data.shape[1]
    temp_dim = temp_data.shape[1]

    balanced_acc = resampled_features[:, :acc_dim]
    balanced_sound = resampled_features[:, acc_dim:acc_dim + sound_dim]
    balanced_temp = resampled_features[:, acc_dim + sound_dim:]

    print(f"平衡后类别分布: {Counter(resampled_labels)}")
    return balanced_acc, balanced_sound, balanced_temp, resampled_labels


def plot_confusion_matrix(cm, class_names, save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def wavelet_denoise(signal, wavelet='db4', level=5, threshold=None):
    """
    对信号进行小波滤噪
    :param signal: 输入信号（如加速度数据）
    :param wavelet: 小波类型，默认使用'db4'
    :param level: 小波分解的层数，默认值为5
    :param threshold: 阈值，用于去噪。若为None，使用默认计算方式
    :return: 滤噪后的信号
    """
    # 小波分解
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # 计算阈值（如果未指定）
    if threshold is None:
        threshold = np.sqrt(2 * np.log(len(signal))) * (1 / 2)  # 基于信号长度和小波分解层数的阈值

    # 阈值去噪（软阈值去噪）
    coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

    # 小波重构
    denoised_signal = pywt.waverec(coeffs_thresholded, wavelet)

    return denoised_signal

def lowpass_filter(data, cutoff=0.1, fs=1.0, order=4):
    """
    应用低通滤波器进行信号滤波
    :param data: 输入信号数据
    :param cutoff: 截止频率
    :param fs: 采样频率
    :param order: 滤波器的阶数
    :return: 滤波后的信号
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def highpass_filter(data, cutoff=0.1, fs=1.0, order=4):
    """
    应用高通滤波器进行信号滤波
    :param data: 输入信号数据
    :param cutoff: 截止频率
    :param fs: 采样频率
    :param order: 滤波器的阶数
    :return: 滤波后的信号
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data