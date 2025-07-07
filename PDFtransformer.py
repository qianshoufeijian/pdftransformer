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

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add new components
class ResidualBlock(nn.Module):
    """Residual block to enhance CNN feature extraction capability"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Residual connection - use 1x1 convolution if dimensions change
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
    """Spatial attention module to enhance image feature extraction"""
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # Calculate mean and max values across the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate features
        x_cat = torch.cat([avg_out, max_out], dim=1)

        # Apply convolution and activation
        attention_map = torch.sigmoid(self.conv1(x_cat))

        # Apply attention map
        return x * attention_map

class ChannelAttention(nn.Module):
    """Channel attention mechanism to focus on important feature channels"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # Average pooling branch
        avg_out = self.fc(self.avg_pool(x).view(b, c))

        # Max pooling branch
        max_out = self.fc(self.max_pool(x).view(b, c))

        # Combine both branches
        attention = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)

        return x * attention

class ModalityToImage:
    """Convert temporal modality data into image representations"""
    def __init__(self, image_size=(224, 224), methods=None, normalize=True):
        """
        Initialize converter
        Parameters:
            image_size: Output image dimensions (height, width)
            methods: Dictionary of conversion methods for each modality {'acc': 'method', 'sound': 'method', 'temp': 'method'}
                     Available methods: 'spectrogram', 'recurrence', 'gramian', 'scalogram', 'raw'
            normalize: Whether to normalize the generated image
        """
        self.image_size = image_size
        self.normalize = normalize

        # Default conversion methods
        default_methods = {
            'acc': 'recurrence',  # Acceleration data fits recurrence plot
            'sound': 'spectrogram',  # Sound data fits spectrogram
            'temp': 'gramian'  # Temperature data fits Gramian matrix
        }

        self.methods = methods if methods is not None else default_methods

        # Image converter
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) if normalize else transforms.Lambda(lambda x: x)
        ])

    def _create_spectrogram(self, data, n_fft=256, hop_length=32):
        """Create spectrogram"""
        try:
            # 1. Data preparation
            data = np.asarray(data).flatten()  # Ensure data is a numpy array and flattened

            # 2. Data length check
            data_len = len(data)
            if data_len == 0:
                # Data is empty, return blank image
                blank_img = np.zeros((128, 128, 3), dtype=np.uint8)
                return Image.fromarray(blank_img)

            # 3. Adaptive parameter setting
            # Ensure nperseg doesn't exceed data length
            nperseg = min(n_fft, data_len)

            # Ensure hop_length is reasonable
            if hop_length >= nperseg:
                # If hop_length is too large, set it to one-fourth of nperseg
                hop_length = max(1, nperseg // 4)

            # Compute noverlap and ensure it's strictly less than nperseg
            noverlap = nperseg - hop_length
            if noverlap >= nperseg:
                noverlap = nperseg - 1

            # 4. Compute Short-Time Fourier Transform (STFT)
            f, t, Zxx = signal.stft(data, nperseg=nperseg, noverlap=noverlap)

            # 5. Process spectrogram - take log of magnitude
            spectrogram = np.log1p(np.abs(Zxx))

            # 6. Normalize to [0,1]
            if np.max(spectrogram) > np.min(spectrogram):
                spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
            else:
                # In case all values are the same
                spectrogram = np.full_like(spectrogram, 0.5)

            # 7. Convert to RGB image
            spectrogram_rgb = np.stack([spectrogram, spectrogram, spectrogram], axis=2)
            spectrogram_rgb = (spectrogram_rgb * 255).astype(np.uint8)

            return Image.fromarray(spectrogram_rgb)

        except Exception as e:
            # Catch all exceptions, return blank image
            blank_img = np.zeros((128, 128, 3), dtype=np.uint8)
            return Image.fromarray(blank_img)

    def _create_recurrence_plot(self, data, embed_dim=10, delay=2, threshold=None):
        """Create recurrence plot"""
        # Ensure data is a one-dimensional array
        data = data.flatten()

        # Adjust embedding dimension if data is too short
        if len(data) < embed_dim * delay:
            embed_dim = max(2, len(data) // delay - 1)
            # Ensure data length is at least embed_dim * delay
        if len(data) < embed_dim * delay:
            # Pad data to sufficient length
            data = np.pad(data, (0, embed_dim * delay - len(data)), mode='constant')

        # Create embedding vectors
        N = len(data) - (embed_dim - 1) * delay
        if N <= 0:  # Simplify handling if data is too short
            embed_dim = 2
            delay = 1
            N = len(data) - (embed_dim - 1) * delay

        vectors = np.zeros((N, embed_dim))
        for i in range(N):
            for j in range(embed_dim):
                vectors[i, j] = data[i + j * delay]

        # Compute distance matrix
        dist = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                d = np.linalg.norm(vectors[i] - vectors[j])
                dist[i, j] = d
                dist[j, i] = d

        # Apply threshold (if specified)
        if threshold is not None:
            if threshold == 'auto':
                threshold = np.mean(dist) + np.std(dist)
            rp = (dist < threshold).astype(np.uint8) * 255
        else:
            # Normalize to [0,255]
            if np.max(dist) > np.min(dist):
                rp = 255 - ((dist - np.min(dist)) / (np.max(dist) - np.min(dist)) * 255).astype(np.uint8)
            else:
                rp = np.zeros_like(dist, dtype=np.uint8)

        # Convert to RGB image
        rp_rgb = np.stack([rp, rp, rp], axis=2)

        return Image.fromarray(rp_rgb)

    def _create_gramian_angular_field(self, data):
        """Create Gramian Angular Field image"""
        # Ensure data is a one-dimensional array
        data = data.flatten()

        # Normalize to [-1,1]
        if np.max(data) > np.min(data):
            scaled_data = 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1
        else:
            scaled_data = np.zeros_like(data)

        # Convert to angles
        phi = np.arccos(scaled_data)

        # Compute Gramian matrix
        n = len(phi)
        gaf = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                gaf[i, j] = np.cos(phi[i] + phi[j])
                gaf[j, i] = gaf[i, j]  # Symmetric matrix

        # Normalize to [0,1]
        gaf = (gaf + 1) / 2

        # Convert to RGB image
        gaf_rgb = np.stack([gaf, gaf, gaf], axis=2)
        gaf_rgb = (gaf_rgb * 255).astype(np.uint8)

        return Image.fromarray(gaf_rgb)

    def _create_scalogram(self, data, scales=None):
        """Create scalogram image using wavelet transform"""
        # Ensure data is a one-dimensional array
        data = data.flatten()

        # Set default scales
        if scales is None:
            scales = np.arange(1, min(128, len(data) // 2))

        # Continuous wavelet transform
        coef, freqs = signal.cwt(data, signal.morlet2, scales)

        # Take log of magnitude
        scalogram = np.log1p(np.abs(coef))

        # Normalize to [0,1]
        if np.max(scalogram) > np.min(scalogram):
            scalogram = (scalogram - np.min(scalogram)) / (np.max(scalogram) - np.min(scalogram))

        # Convert to RGB image
        scalogram_rgb = np.stack([scalogram, scalogram, scalogram], axis=2)
        scalogram_rgb = (scalogram_rgb * 255).astype(np.uint8)

        return Image.fromarray(scalogram_rgb)

    def _create_raw_image(self, data, n_rows=None):
        """Reshape raw data to 2D image"""
        # Ensure data is a one-dimensional array
        data = data.flatten()

        # Determine number of rows and columns
        if n_rows is None:
            n_rows = int(np.sqrt(len(data)))

        n_cols = int(np.ceil(len(data) / n_rows))

        # Pad data to match row and column count
        padded_data = np.zeros(n_rows * n_cols)
        padded_data[:len(data)] = data

        # Reshape to 2D
        img_data = padded_data.reshape(n_rows, n_cols)

        # Normalize to [0,1]
        if np.max(img_data) > np.min(img_data):
            img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

        # Convert to RGB image
        img_rgb = np.stack([img_data, img_data, img_data], axis=2)
        img_rgb = (img_rgb * 255).astype(np.uint8)

        return Image.fromarray(img_rgb)

    def convert(self, data, modality):
        """
        Convert single modality data to image
        Parameters:
            data: Input data
            modality: Modality type ('acc', 'sound', 'temp')

        Returns:
            Converted image tensor
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

            # Apply transformation
            return self.transform(img)

        except Exception as e:
            print(f"Error converting {modality} data: {e}")
            # Return blank image on error
            blank = np.zeros((*self.image_size, 3), dtype=np.uint8)
            return self.transform(Image.fromarray(blank))

class CustomDataset(Dataset):
    def __init__(self, acc_data, sound_data, temp_data, labels, transform=False, scalers=None,
                 aug_strength=0.8, convert_to_image=False, image_size=(224, 224)):
        self.acc_data = acc_data
        self.sound_data = sound_data
        self.temp_data = temp_data
        self.labels = labels
        self.transform = transform
        self.aug_strength = aug_strength  # Control augmentation strength
        self.convert_to_image = convert_to_image

        # If needed, create converter to image
        if self.convert_to_image:
            # Configure the most suitable conversion method for different modalities
            image_methods = {
                'acc': 'recurrence',  # Acceleration data fits recurrence plot
                'sound': 'spectrogram',  # Sound data fits spectrogram
                'temp': 'gramian'  # Temperature data fits Gramian matrix
            }
            self.image_converter = ModalityToImage(
                image_size=image_size,
                methods=image_methods,
                normalize=True
            )

            # Skip normalization
            self.scalers = None

            # Enhanced image transformations - more powerful data augmentation
            if self.transform:
                self.image_transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),  # Increase rotation angle
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),  # More color perturbations
                    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),  # Add scaling
                    transforms.RandomPerspective(distortion_scale=0.15, p=0.5),  # Enhance perspective transformation
                    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.3),  # Add Gaussian blur
                    transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),  # Add sharpening
                ])
            else:
                self.image_transforms = None
        else:
            # Standardization processing
            if scalers is None:
                # Create and fit scaler on training set
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
                # Use training set's scaler on validation set
                self.acc_data = scalers['acc'].transform(self.acc_data)
                self.sound_data = scalers['sound'].transform(self.sound_data)
                self.temp_data = scalers['temp'].transform(self.temp_data)
                self.scalers = scalers

    def __len__(self):
        return len(self.labels)

    def _time_warp(self, x, sigma=0.2):
        """Time warp augmentation - more realistic than simple flipping"""
        if x.shape[0] <= 3:  # Skip if sequence is too short
            return x
        orig_steps = np.arange(x.shape[0])
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=x.shape[0])
        warp_steps = np.cumsum(random_warps)
        warp_steps = warp_steps / warp_steps[-1] * (x.shape[0] - 1)
        warped_x = np.interp(orig_steps, warp_steps, x)
        return warped_x

    def _freq_mask(self, x, num_masks=1, width_range=(0.05, 0.15)):
        """Frequency domain mask - simulate sensor failure"""
        x_aug = x.copy()
        for i in range(num_masks):
            mask_width = int(x.shape[0] * np.random.uniform(*width_range))
            if mask_width > 0:
                mask_start = np.random.randint(0, x.shape[0] - mask_width)
                x_aug[mask_start:mask_start + mask_width] = 0
        return x_aug

    def _random_scaling(self, x, scale_range=(0.8, 1.2)):
        """Apply random scaling to simulate different sensor sensitivity"""
        scale = np.random.uniform(*scale_range)
        return x * scale

    def _magnitude_warp(self, x, sigma=0.2):
        """Magnitude warp, create random smooth change curve"""
        if x.shape[0] <= 3:
            return x

        # Create smooth curve to adjust magnitude
        knot_points = max(3, int(x.shape[0] / 10))
        knots = np.random.normal(loc=1.0, scale=sigma, size=knot_points)
        knot_positions = np.linspace(0, x.shape[0] - 1, knot_points)

        # Interpolate to generate smooth curve
        magnitude_changes = np.interp(np.arange(x.shape[0]), knot_positions, knots)

        return x * magnitude_changes

    def _jitter(self, x, sigma=0.05):
        """Add jitter noise to each point individually"""
        return x + np.random.normal(0, sigma, x.shape)

    def _window_slice(self, x, reduce_ratio=0.9):
        """Extract continuous slice of signal and interpolate back to original length"""
        if x.shape[0] <= 3:
            return x

        new_len = int(x.shape[0] * reduce_ratio)
        if new_len <= 1:
            return x

        start_idx = np.random.randint(0, x.shape[0] - new_len + 1)
        sliced = x[start_idx:start_idx + new_len]

        # Adjust back to original length
        return np.interp(
            np.linspace(0, sliced.shape[0] - 1, x.shape[0]),
            np.arange(sliced.shape[0]),
            sliced
        )

    def _trend_injection(self, x):
        """Add random trend to signal"""
        trend_strength = np.random.uniform(0.1, 0.3) * self.aug_strength
        trend = np.linspace(0, 1, x.shape[0]) * trend_strength * np.random.choice([-1, 1])
        return x + trend

    def __getitem__(self, idx):
        if self.convert_to_image:
            # Get raw data
            acc = self.acc_data[idx].copy()
            sound = self.sound_data[idx].copy()
            temp = self.temp_data[idx].copy()
            label = self.labels[idx]

            # Convert to image representation
            acc_img = self.image_converter.convert(acc, 'acc')
            sound_img = self.image_converter.convert(sound, 'sound')
            temp_img = self.image_converter.convert(temp, 'temp')

            # Apply image augmentation (if enabled)
            if self.transform and self.image_transforms is not None:
                # Generate random seed for consistency
                seed = torch.randint(0, 2 ** 32, (1,)).item()

                # Apply same random transformation to each modality
                torch.manual_seed(seed)
                acc_img = self.image_transforms(acc_img)

                torch.manual_seed(seed)
                sound_img = self.image_transforms(sound_img)

                torch.manual_seed(seed)
                temp_img = self.image_transforms(temp_img)
                # There's a 10% chance to randomly replace one modality with noise to enhance robustness
                if np.random.random() < 0.1 * self.aug_strength:
                    noise_idx = np.random.randint(0, 3)
                    noise_img = torch.randn_like(acc_img) * 0.1  # Low intensity noise
                    if noise_idx == 0:
                        acc_img = (acc_img * 0.8) + noise_img
                    elif noise_idx == 1:
                        sound_img = (sound_img * 0.8) + noise_img
                    else:
                        temp_img = (temp_img * 0.8) + noise_img
            return acc_img, sound_img, temp_img, torch.tensor(label, dtype=torch.long)

        else:
            acc = self.acc_data[idx].copy()
            sound = self.sound_data[idx].copy()
            temp = self.temp_data[idx].copy()
            label = self.labels[idx]

            if self.transform:
                # Set different random seeds to avoid inter-modality enhancement correlations
                np.random.seed(int(idx * 1000 + np.random.randint(1000)))

                # 1. Advanced noise addition - different noise levels
                if np.random.random() < self.aug_strength * 0.7:
                    noise_level = np.random.uniform(0.01, 0.15 * self.aug_strength)
                    acc = acc + np.random.normal(0, noise_level, acc.shape)

                if np.random.random() < self.aug_strength * 0.7:
                    noise_level = np.random.uniform(0.01, 0.15 * self.aug_strength)
                    sound = sound + np.random.normal(0, noise_level, sound.shape)

                if np.random.random() < self.aug_strength * 0.5:
                    noise_level = np.random.uniform(0.005, 0.05 * self.aug_strength)
                    temp = temp + np.random.normal(0, noise_level, temp.shape)

                # 2. Time warp - more realistic than simple flipping
                if np.random.random() < self.aug_strength * 0.5:
                    acc = self._time_warp(acc, sigma=0.2 * self.aug_strength)

                if np.random.random() < self.aug_strength * 0.5:
                    sound = self._time_warp(sound, sigma=0.2 * self.aug_strength)

                if np.random.random() < self.aug_strength * 0.3:
                    temp = self._time_warp(temp, sigma=0.1 * self.aug_strength)

                # 3. Frequency domain mask - simulate sensor failure
                if np.random.random() < self.aug_strength * 0.6:
                    acc = self._freq_mask(acc, num_masks=int(1 + self.aug_strength * 2),
                                          width_range=(0.05, 0.2 * self.aug_strength))

                if np.random.random() < self.aug_strength * 0.6:
                    sound = self._freq_mask(sound, num_masks=int(1 + self.aug_strength * 2),
                                            width_range=(0.05, 0.2 * self.aug_strength))

                # 4. Random scaling - simulate different sensor sensitivity
                if np.random.random() < self.aug_strength * 0.5:
                    acc = self._random_scaling(acc,
                                               scale_range=(1.0 - 0.3 * self.aug_strength,
                                                            1.0 + 0.3 * self.aug_strength))

                if np.random.random() < self.aug_strength * 0.5:
                    sound = self._random_scaling(sound,
                                                 scale_range=(1.0 - 0.3 * self.aug_strength,
                                                              1.0 + 0.3 * self.aug_strength))

                # 5. Magnitude warp - newly added
                if np.random.random() < self.aug_strength * 0.5:
                    acc = self._magnitude_warp(acc, sigma=0.2 * self.aug_strength)

                if np.random.random() < self.aug_strength * 0.5:
                    sound = self._magnitude_warp(sound, sigma=0.2 * self.aug_strength)

                # 6. Jitter - newly added
                if np.random.random() < self.aug_strength * 0.5:
                    acc = self._jitter(acc, sigma=0.05 * self.aug_strength)

                if np.random.random() < self.aug_strength * 0.5:
                    sound = self._jitter(sound, sigma=0.05 * self.aug_strength)

                # 7. Window slicing - newly added
                if np.random.random() < self.aug_strength * 0.4:
                    slice_ratio = np.random.uniform(0.85, 0.97)
                    acc = self._window_slice(acc, reduce_ratio=slice_ratio)
                    sound = self._window_slice(sound, reduce_ratio=slice_ratio)
                    temp = self._window_slice(temp, reduce_ratio=slice_ratio)

                # 8. Trend injection - newly added
                if np.random.random() < self.aug_strength * 0.3:
                    acc = self._trend_injection(acc)

                if np.random.random() < self.aug_strength * 0.3:
                    temp = self._trend_injection(temp)

                # 9. Flip enhancement (lower probability)
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
        weights = F.softplus(self.fc3(x))  # Use softplus to ensure weights are positive and smooth
        return weights

class MultiModalFusion(nn.Module):
    def __init__(self, embed_dim, num_patches, num_heads, num_fusion_layers=4):  # Increased from 3 to 4
        super(MultiModalFusion, self).__init__()
        self.embed_dim = embed_dim  # Save as class attribute
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)

        # Add cross-modal feature enhancement layer
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

        # Attention mechanism specific to image modality
        self.image_attention = nn.ModuleDict({
            'acc': nn.MultiheadAttention(embed_dim, num_heads // 2, dropout=0.1),
            'sound': nn.MultiheadAttention(embed_dim, num_heads // 2, dropout=0.1),
            'temp': nn.MultiheadAttention(embed_dim, num_heads // 2, dropout=0.1)
        })

        # Multi-layer fusion network
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

        # Enhanced inter-modal pair attention mechanism - add reverse direction
        self.cross_modal_attention = nn.ModuleDict({
            'acc_sound': nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1),
            'acc_temp': nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1),
            'sound_temp': nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1),
            'sound_acc': nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1),  # Newly added
            'temp_acc': nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1),  # Newly added
            'temp_sound': nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)  # Newly added
        })

        self.cross_norms = nn.ModuleDict({
            'acc': nn.LayerNorm(embed_dim),
            'sound': nn.LayerNorm(embed_dim),
            'temp': nn.LayerNorm(embed_dim)
        })

        # Add gated fusion mechanism
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

        # Initially apply modality-specific transformations
        acc_patches = self.modal_fc['acc'](acc_patches)
        sound_patches = self.modal_fc['sound'](sound_patches)
        temp_patches = self.modal_fc['temp'](temp_patches)

        # Enhanced feature representation
        acc_enhanced = self.modal_enhance['acc'](acc_patches)
        sound_enhanced = self.modal_enhance['sound'](sound_patches)
        temp_enhanced = self.modal_enhance['temp'](temp_patches)

        # Apply cross-patch self-attention - designed specifically for image features
        acc_t = acc_enhanced.transpose(0, 1)
        sound_t = sound_enhanced.transpose(0, 1)
        temp_t = temp_enhanced.transpose(0, 1)

        acc_self, _ = self.image_attention['acc'](acc_t, acc_t, acc_t)
        sound_self, _ = self.image_attention['sound'](sound_t, sound_t, sound_t)
        temp_self, _ = self.image_attention['temp'](temp_t, temp_t, temp_t)

        # Residual connections of original and enhanced features
        acc_t = acc_t + 0.2 * acc_self
        sound_t = sound_t + 0.2 * sound_self
        temp_t = temp_t + 0.2 * temp_self

        # Add CLS token and positional encoding
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        acc_input = torch.cat([cls_token, acc_patches], dim=1) + self.pos_embed
        sound_input = torch.cat([cls_token, sound_patches], dim=1) + self.pos_embed
        temp_input = torch.cat([cls_token, temp_patches], dim=1) + self.pos_embed

        # Apply dynamic weights and add residual connections
        acc_input = acc_input * (1.0 + acc_weights.unsqueeze(1))
        sound_input = sound_input * (1.0 + sound_weights.unsqueeze(1))
        temp_input = temp_input * (1.0 + temp_weights.unsqueeze(1))

        # Transpose to format required by attention layer (seq_len, batch, features)
        acc_t = acc_input.transpose(0, 1)
        sound_t = sound_input.transpose(0, 1)
        temp_t = temp_input.transpose(0, 1)

        # Bi-directional inter-modal attention calculation
        acc_sound, _ = self.cross_modal_attention['acc_sound'](acc_t, sound_t, sound_t)
        sound_acc, _ = self.cross_modal_attention['sound_acc'](sound_t, acc_t, acc_t)
        acc_temp, _ = self.cross_modal_attention['acc_temp'](acc_t, temp_t, temp_t)
        temp_acc, _ = self.cross_modal_attention['temp_acc'](temp_t, acc_t, acc_t)
        sound_temp, _ = self.cross_modal_attention['sound_temp'](sound_t, temp_t, temp_t)
        temp_sound, _ = self.cross_modal_attention['temp_sound'](temp_t, sound_t, sound_t)

        # Apply residual connection and layer normalization, fuse bi-directional information
        acc_t = self.cross_norms['acc'](acc_t + acc_sound + acc_temp)
        sound_t = self.cross_norms['sound'](sound_t + sound_acc + sound_temp)
        temp_t = self.cross_norms['temp'](temp_t + temp_acc + temp_sound)

        # Implement dynamic gated fusion
        combined = torch.cat([
            acc_t.transpose(0, 1).mean(dim=1),
            sound_t.transpose(0, 1).mean(dim=1),
            temp_t.transpose(0, 1).mean(dim=1)
        ], dim=1)

        gate_outputs = self.fusion_gates(combined)  # (batch_size, 3 * embed_dim)
        gate_chunks = gate_outputs.chunk(3, dim=1)  # Three tensors of shape (batch_size, embed_dim)
        acc_gate, sound_gate, temp_gate = gate_chunks

        # Apply gating mechanism
        acc_gate = acc_gate.unsqueeze(0)  # (1, batch_size, embed_dim)
        sound_gate = sound_gate.unsqueeze(0)  # (1, batch_size, embed_dim)
        temp_gate = temp_gate.unsqueeze(0)  # (1, batch_size, embed_dim)
        acc_t = acc_t * acc_gate
        sound_t = sound_t * sound_gate
        temp_t = temp_t * temp_gate

        # Multi-layer fusion and residual connection
        fused = torch.cat([acc_t, sound_t, temp_t], dim=0)  # Concatenate along sequence dimension
        residual_fused = fused  # For residual connection

        for layer in self.fusion_layers:
            # Multi-head attention
            attn_out, _ = layer['attn'](fused, fused, fused)
            fused = layer['norm1'](fused + attn_out)

            # Feed-forward network
            ffn_out = layer['ffn'](fused)
            fused = layer['norm2'](fused + ffn_out)

            # Add scaled residual connection to prevent gradient vanishing
            fused = fused + residual_fused * 0.1

        # Return to batch-first format
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
                nn.Linear(input_shape[0], projection_dim * 2),
                nn.LayerNorm(projection_dim * 2),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(projection_dim * 2, projection_dim * 2),
                nn.LayerNorm(projection_dim * 2),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(projection_dim * 2, projection_dim * num_patches),
                nn.Unflatten(1, (num_patches, projection_dim))
            )

            # Similar encoders for sound and temp with same structure
            self.sound_encoder = self._build_similar_encoder(input_shape[1])
            self.temp_encoder = self._build_similar_encoder(input_shape[2])

        # Reduce number of transformer layers for stability - from 3 to 2
        self.acc_transformer = nn.ModuleList([
            self._build_transformer_layer() for _ in range(2)
        ])
        self.sound_transformer = nn.ModuleList([
            self._build_transformer_layer() for _ in range(2)
        ])
        self.temp_transformer = nn.ModuleList([
            self._build_transformer_layer() for _ in range(2)
        ])

        # Simplified fusion layer with proper initialization
        self.fusion_layer = self._build_simplified_fusion()

        # Dynamic weight prediction with proper initialization
        self.acc_prediction_module = PredictionModule(input_dim=projection_dim, output_dim=1)
        self.sound_prediction_module = PredictionModule(input_dim=projection_dim, output_dim=1)
        self.temp_prediction_module = PredictionModule(input_dim=projection_dim, output_dim=1)

        # Reduce main transformer layers - from 8 to 6 for better stability
        self.transformer_layers = nn.ModuleList([
            self._build_transformer_layer() for _ in range(6)
        ])

        # Enhanced classifiers
        self.mid_classifier = self._build_enhanced_classifier(num_classes)
        self.acc_classifier = self._build_enhanced_classifier(num_classes)
        self.sound_classifier = self._build_enhanced_classifier(num_classes)
        self.temp_classifier = self._build_enhanced_classifier(num_classes)
        self.fc_out = self._build_enhanced_output_classifier(num_classes)

        # Initialize weights properly
        self._initialize_weights()

    def _build_similar_encoder(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, self.projection_dim * 2),
            nn.LayerNorm(self.projection_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.projection_dim * 2, self.projection_dim * 2),
            nn.LayerNorm(self.projection_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.projection_dim * 2, self.projection_dim * self.num_patches),
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
        """Build enhanced CNN encoder to handle image input"""
        return nn.Sequential(
            # First layer - initial feature extraction
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # [B, 64, 112, 112]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [B, 64, 56, 56]

            # Residual block 1
            ResidualBlock(64, 128, stride=2),  # [B, 128, 28, 28]
            SpatialAttention(128),  # Add spatial attention

            # Residual block 2
            ResidualBlock(128, 256, stride=2),  # [B, 256, 14, 14]
            SpatialAttention(256),  # Add spatial attention

            # Residual block 3
            ResidualBlock(256, 512, stride=2),  # [B, 512, 7, 7]

            # Channel attention
            ChannelAttention(512),

            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 512, 1, 1]
            nn.Flatten(),  # [B, 512]

            # Map to required dimension
            nn.Linear(512, self.projection_dim * 2),
            nn.LayerNorm(self.projection_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.projection_dim * 2, self.projection_dim * self.num_patches),
            nn.Unflatten(1, (self.num_patches, self.projection_dim))  # [B, num_patches, projection_dim]
        )

    def _build_cnn_encoder(self):
        """Build CNN encoder to handle image input"""
        return self._build_enhanced_cnn_encoder()  # Use the enhanced encoder

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
        # Encode each modality
        if self.image_input:
            # If input is images, pass directly to CNN encoder
            acc_patches = self.acc_encoder(acc_data)
            sound_patches = self.sound_encoder(sound_data)
            temp_patches = self.temp_encoder(temp_data)
        else:
            # Encode raw temporal data
            acc_patches = self.acc_encoder(acc_data)
            sound_patches = self.sound_encoder(sound_data)
            temp_patches = self.temp_encoder(temp_data)

        # Save original patches for residual connection
        acc_original = acc_patches
        sound_original = sound_patches
        temp_original = temp_patches

        # Apply modality-specific transformers to learn better representations and add residual connection
        for layer in self.acc_transformer:
            acc_patches = layer(acc_patches)
            acc_patches = acc_patches + 0.1 * acc_original  # Add residual connection

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

        # Modality dynamic weight calculation
        acc_weights = self.acc_prediction_module(acc_feat)
        sound_weights = self.sound_prediction_module(sound_feat)
        temp_weights = self.temp_prediction_module(temp_feat)

        # Improved weight normalization - use softmax rather than simple division
        weights = torch.cat([acc_weights, sound_weights, temp_weights], dim=1)
        normalized_weights = F.softmax(weights, dim=1)
        acc_weights = normalized_weights[:, 0].unsqueeze(1)
        sound_weights = normalized_weights[:, 1].unsqueeze(1)
        temp_weights = normalized_weights[:, 2].unsqueeze(1)

        # Improved fusion
        fused_features = self.fusion_layer(
            acc_patches, sound_patches, temp_patches,
            acc_weights, sound_weights, temp_weights
        )

        # Main transformer processing and multi-point supervision
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

        # Main classification from CLS token
        main_logits = self.fc_out(x[:, 0])

        # Weighted prediction combination (dynamic ensemble)
        if self.training:
            # During training, use all predictions and give higher weights to auxiliary predictions
            final_logits = (main_logits * 0.6 +
                            mid_logits * 0.15 +
                            acc_pred * 0.1 +
                            sound_pred * 0.1 +
                            temp_pred * 0.05)
        else:
            # During inference, focus more on the main classifier
            final_logits = (main_logits * 0.7 +
                            mid_logits * 0.1 +
                            acc_pred * 0.08 +
                            sound_pred * 0.08 +
                            temp_pred * 0.04)

        return final_logits

class EnhancedMixedLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1, temp=1.0, reduction='mean'):
        super(EnhancedMixedLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focal loss gamma
        self.smoothing = smoothing  # Label smoothing factor
        self.temp = temp  # Temperature for softening probability distributions
        self.reduction = reduction
        self.num_classes = len(alpha) if alpha is not None else None

        # Add support for dynamic focusing
        self.dynamic_focusing = True

    def forward(self, inputs, targets):
        # 确保targets是合适的形状
        if targets.dim() > 1 and targets.size(1) > 1:
            # 如果targets是多维的，取argmax将其转换为1D
            _, targets = torch.max(targets, dim=1)

        # 确保targets是长整型
        targets = targets.long()

        # 应用温度缩放
        scaled_inputs = inputs / self.temp

        # 计算标准交叉熵
        try:
            ce_loss = F.cross_entropy(scaled_inputs, targets, weight=self.alpha, reduction='none')
        except RuntimeError:
            print(f"错误：inputs shape: {inputs.shape}， targets shape: {targets.shape}")
            # 如果还是有错误，采用简单的交叉熵
            return F.cross_entropy(inputs, targets.reshape(-1), weight=self.alpha, reduction=self.reduction)

        # 获取softmax概率
        probs = F.softmax(scaled_inputs, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        # Apply dynamic focusing based on sample difficulty
        if self.dynamic_focusing:
            # Hard samples get higher gamma
            sample_difficulty = 1 - pt
            batch_gamma = self.gamma + sample_difficulty * 2.0
            focal_weight = (1 - pt) ** batch_gamma
        else:
            # Standard focal weight
            focal_weight = (1 - pt) ** self.gamma

        # Apply focal weighting
        focal_loss = focal_weight * ce_loss

        # Apply label smoothing if enabled
        if self.smoothing > 0 and self.num_classes is not None:
            # Create smoothed targets
            smooth_targets = torch.zeros_like(scaled_inputs).scatter_(
                1, targets.unsqueeze(1), 1.0)
            smooth_targets = smooth_targets * (1 - self.smoothing) + self.smoothing / self.num_classes

            # Compute KL-divergence loss for smoothed targets
            log_probs = F.log_softmax(scaled_inputs, dim=1)
            smooth_loss = -(smooth_targets * log_probs).sum(dim=1)

            # Combine losses - less smoothing for difficult samples
            smooth_weight = torch.exp(-focal_weight * 0.5)
            combined_loss = focal_loss * 0.8 + smooth_loss * 0.2 * smooth_weight
        else:
            combined_loss = focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return combined_loss.mean()
        elif self.reduction == 'sum':
            return combined_loss.sum()
        else:
            return combined_loss

def get_enhanced_lr_scheduler(optimizer, args, train_loader=None):
    if args.lr_scheduler == 'cosine_warm':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2)  # Faster restart cycle
    elif args.lr_scheduler == 'one_cycle':
        if train_loader is None:
            raise ValueError("train_loader must be provided for one_cycle scheduler")
        return optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr * 15,  # Higher peak learning rate
            total_steps=args.epochs * len(train_loader),
            pct_start=0.3,  # Faster warm-up
            div_factor=25.0,  # Larger initial learning rate divisor
            final_div_factor=1000.0)  # Smaller final learning rate
    elif args.lr_scheduler == 'cosine_warmup':
        # Custom warm-up + cosine annealing scheduler
        warmup_epochs = max(3, int(args.epochs * 0.1))

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif args.lr_scheduler == 'cyclic':
        # More aggressive cyclic learning rate
        step_size_up = len(train_loader) * 2
        return optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=args.lr / 10, max_lr=args.lr * 10,
            step_size_up=step_size_up, mode='triangular2')
    elif args.lr_scheduler == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3,  # Reduce patience value
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

def balance_dataset(acc_data, sound_data, temp_data, labels, sampling_strategy='auto'):
    print(f"Original class distribution: {Counter(labels)}")

    # Combine features for SMOTE
    combined_features = np.hstack([acc_data, sound_data, temp_data])

    # Apply SMOTE oversampling
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    resampled_features, resampled_labels = smote.fit_resample(combined_features, labels)

    # Separate features of different modalities
    acc_dim = acc_data.shape[1]
    sound_dim = sound_data.shape[1]
    temp_dim = temp_data.shape[1]

    balanced_acc = resampled_features[:, :acc_dim]
    balanced_sound = resampled_features[:, acc_dim:acc_dim + sound_dim]
    balanced_temp = resampled_features[:, acc_dim + sound_dim:]

    print(f"Balanced class distribution: {Counter(resampled_labels)}")

    return balanced_acc, balanced_sound, balanced_temp, resampled_labels

def plot_confusion_matrix(cm, class_names, save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
