import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
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
            methods: Dictionary of conversion methods for each modality {'force_x': 'method', 'vib_x': 'method', 'sound': 'method'}
                     Available methods: 'spectrogram', 'recurrence', 'gramian', 'scalogram', 'raw'
            normalize: Whether to normalize the generated image
        """
        self.image_size = image_size
        self.normalize = normalize

        # Default conversion methods
        default_methods = {
            'force_x': 'recurrence',  # Force signals fit recurrence plot
            'force_y': 'recurrence',
            'force_z': 'recurrence',
            'vib_x': 'recurrence',  # Vibration signals fit recurrence plot
            'vib_y': 'recurrence',
            'vib_z': 'recurrence',
            'sound': 'spectrogram'  # Sound data fits spectrogram
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
            modality: Modality type ('force_x', 'vib_y', 'sound', etc.)

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


def clip_and_scale_force(data, scaler):
    """Special normalization preprocessing for force signals (N)"""
    data_flat = data.reshape(data.shape[0], -1)
    # Force signals typically have larger amplitudes, possibly with sudden peaks, use IQR to handle outliers
    q1, q3 = np.percentile(data_flat, [25, 75], axis=0)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data_clipped = np.clip(data_flat, lower_bound, upper_bound)
    # Use RobustScaler for force signals for more robust scaling
    if not isinstance(scaler, RobustScaler):
        scaler = RobustScaler()
    return scaler.fit_transform(data_clipped).reshape(data.shape)


def clip_and_scale_vibration(data, scaler):
    """Special normalization preprocessing for vibration signals (g)"""
    data_flat = data.reshape(data.shape[0], -1)
    # Vibration signals may need to preserve more subtle variations
    mean = np.mean(data_flat, axis=0)
    std = np.std(data_flat, axis=0)
    # Use 4sigma instead of 3sigma to retain more details
    data_clipped = np.clip(data_flat, mean - 4 * std, mean + 4 * std)
    # Use StandardScaler for vibration signals
    if not isinstance(scaler, StandardScaler):
        scaler = StandardScaler()
    return scaler.fit_transform(data_clipped).reshape(data.shape)


def clip_and_scale_sound(data, scaler):
    """Special normalization preprocessing for sound signals (V)"""
    data_flat = data.reshape(data.shape[0], -1)
    # Sound signals can have sudden peaks, MinMaxScaler is better for preserving relative intensity
    if not isinstance(scaler, MinMaxScaler):
        scaler = MinMaxScaler(feature_range=(-1, 1))
    # Apply log transformation first to handle different amplitude levels
    data_log = np.sign(data_flat) * np.log1p(np.abs(data_flat))
    return scaler.fit_transform(data_log).reshape(data.shape)


class CustomDataset(Dataset):
    def __init__(self, force_x, force_y, force_z, vib_x, vib_y, vib_z, sound, labels,
                 transform=False, scalers=None, aug_strength=0.8, convert_to_image=False, image_size=(224, 224)):
        self.force_x = force_x.copy()
        self.force_y = force_y.copy()
        self.force_z = force_z.copy()
        self.vib_x = vib_x.copy()
        self.vib_y = vib_y.copy()
        self.vib_z = vib_z.copy()
        self.sound = sound.copy()
        self.labels = labels
        self.transform = transform
        self.aug_strength = aug_strength  # Control augmentation strength
        self.convert_to_image = convert_to_image

        # If needed, create converter to image
        if self.convert_to_image:
            # Configure the most suitable conversion method for different modalities
            image_methods = {
                'force_x': 'recurrence',  # Force signals fit recurrence plots
                'force_y': 'recurrence',
                'force_z': 'recurrence',
                'vib_x': 'recurrence',  # Vibration signals fit recurrence plots
                'vib_y': 'recurrence',
                'vib_z': 'recurrence',
                'sound': 'spectrogram'  # Sound signal fits spectrogram
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
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
                    # More color perturbations
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
                # Use different types of standardization for different physical quantities
                self.force_x_scaler = RobustScaler()  # Force signal (N) with RobustScaler
                self.force_y_scaler = RobustScaler()
                self.force_z_scaler = RobustScaler()

                self.vib_x_scaler = StandardScaler()  # Vibration signal (g) with StandardScaler
                self.vib_y_scaler = StandardScaler()
                self.vib_z_scaler = StandardScaler()

                self.sound_scaler = MinMaxScaler(feature_range=(-1, 1))  # Sound signal (V) with MinMaxScaler

                # Apply specialized normalization strategy based on different physical quantities
                print(
                    "Applying specialized normalization for different physical units (Force: N, Vibration: g, Sound: V)")

                # Apply force signal (N) specific normalization
                self.force_x = clip_and_scale_force(self.force_x, self.force_x_scaler)
                self.force_y = clip_and_scale_force(self.force_y, self.force_y_scaler)
                self.force_z = clip_and_scale_force(self.force_z, self.force_z_scaler)

                # Apply vibration signal (g) specific normalization
                self.vib_x = clip_and_scale_vibration(self.vib_x, self.vib_x_scaler)
                self.vib_y = clip_and_scale_vibration(self.vib_y, self.vib_y_scaler)
                self.vib_z = clip_and_scale_vibration(self.vib_z, self.vib_z_scaler)

                # Apply sound signal (V) specific normalization
                self.sound = clip_and_scale_sound(self.sound, self.sound_scaler)

                self.scalers = {
                    'force_x': self.force_x_scaler,
                'force_y': self.force_y_scaler,
                'force_z': self.force_z_scaler,
                'vib_x': self.vib_x_scaler,
                'vib_y': self.vib_y_scaler,
                'vib_z': self.vib_z_scaler,
                'sound': self.sound_scaler
                }

                # Print statistics after normalization to verify normalization effect
                print(
                    f"Force X stats: mean={np.mean(self.force_x):.4f}, std={np.std(self.force_x):.4f}, min={np.min(self.force_x):.4f}, max={np.max(self.force_x):.4f}")
                print(
                    f"Vib X stats: mean={np.mean(self.vib_x):.4f}, std={np.std(self.vib_x):.4f}, min={np.min(self.vib_x):.4f}, max={np.max(self.vib_x):.4f}")
                print(
                    f"Sound stats: mean={np.mean(self.sound):.4f}, std={np.std(self.sound):.4f}, min={np.min(self.sound):.4f}, max={np.max(self.sound):.4f}")

            else:
                # Use the training set's scaler for the validation set
                self.force_x = scalers['force_x'].transform(self.force_x.reshape(self.force_x.shape[0], -1)).reshape(
                    self.force_x.shape)
                self.force_y = scalers['force_y'].transform(self.force_y.reshape(self.force_y.shape[0], -1)).reshape(
                    self.force_y.shape)
                self.force_z = scalers['force_z'].transform(self.force_z.reshape(self.force_z.shape[0], -1)).reshape(
                    self.force_z.shape)

                self.vib_x = scalers['vib_x'].transform(self.vib_x.reshape(self.vib_x.shape[0], -1)).reshape(
                    self.vib_x.shape)
                self.vib_y = scalers['vib_y'].transform(self.vib_y.reshape(self.vib_y.shape[0], -1)).reshape(
                    self.vib_y.shape)
                self.vib_z = scalers['vib_z'].transform(self.vib_z.reshape(self.vib_z.shape[0], -1)).reshape(
                    self.vib_z.shape)

                # Apply log transformation before applying MinMaxScaler for sound signal
                sound_data_log = np.sign(self.sound) * np.log1p(np.abs(self.sound))
                self.sound = scalers['sound'].transform(sound_data_log.reshape(self.sound.shape[0], -1)).reshape(
                    self.sound.shape)

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

    def _trend_injection(self, x, strength_range=(0.1, 0.3)):
        """Add random trend to signal"""
        trend_strength = np.random.uniform(*strength_range) * self.aug_strength
        trend = np.linspace(0, 1, x.shape[0]) * trend_strength * np.random.choice([-1, 1])
        return x + trend

    def _augment_force_signal(self, x):
        """Force signal (N) specific enhancement"""
        if np.random.random() < self.aug_strength * 0.7:
            # Force signals typically represent applied external forces, noise should be smaller
            noise_level = np.random.uniform(0.005, 0.08 * self.aug_strength)
            x = x + np.random.normal(0, noise_level, x.shape)

        if np.random.random() < self.aug_strength * 0.5:
            # Force signal trend disturbance, simulating gradually applying or reducing force
            x = self._trend_injection(x, strength_range=(0.05, 0.12))

        if np.random.random() < self.aug_strength * 0.4:
            # Force signal amplitude disturbance, simulating unstable force changes
            x = self._magnitude_warp(x, sigma=0.12 * self.aug_strength)

        return x

    def _augment_vibration_signal(self, x):
        """Vibration signal (g) specific enhancement"""
        if np.random.random() < self.aug_strength * 0.7:
            # Vibration signals are usually more sensitive, can accept more noise disturbance
            noise_level = np.random.uniform(0.01, 0.15 * self.aug_strength)
            x = x + np.random.normal(0, noise_level, x.shape)

        if np.random.random() < self.aug_strength * 0.6:
            # Frequency domain mask, simulating sensor frequency response changes or environmental interference
            x = self._freq_mask(x, num_masks=int(1 + self.aug_strength * 2),
                                width_range=(0.05, 0.18 * self.aug_strength))

        if np.random.random() < self.aug_strength * 0.5:
            # Time warp, simulating uneven vibration sampling times
            x = self._time_warp(x, sigma=0.2 * self.aug_strength)

        return x

    def _augment_sound_signal(self, x):
        """Sound signal (V) specific enhancement"""
        if np.random.random() < self.aug_strength * 0.8:
            # Sound signals can accept greater noise interference, simulating environmental noise
            noise_level = np.random.uniform(0.01, 0.2 * self.aug_strength)
            x = x + np.random.normal(0, noise_level, x.shape)

        if np.random.random() < self.aug_strength * 0.6:
            # Random scaling, simulating different recording volumes or gain settings
            x = self._random_scaling(x, scale_range=(1.0 - 0.25 * self.aug_strength,
                                                     1.0 + 0.25 * self.aug_strength))

        if np.random.random() < self.aug_strength * 0.7:
            # Frequency domain mask, simulating frequency selective attenuation or loss
            x = self._freq_mask(x, num_masks=int(1 + self.aug_strength * 3),
                                width_range=(0.05, 0.22 * self.aug_strength))

        return x

    def __getitem__(self, idx):
        if self.convert_to_image:
            # Get raw data
            fx = self.force_x[idx].copy()
            fy = self.force_y[idx].copy()
            fz = self.force_z[idx].copy()
            vx = self.vib_x[idx].copy()
            vy = self.vib_y[idx].copy()
            vz = self.vib_z[idx].copy()
            sound = self.sound[idx].copy()
            label = self.labels[idx]

            # Convert to image representation
            fx_img = self.image_converter.convert(fx, 'force_x')
            fy_img = self.image_converter.convert(fy, 'force_y')
            fz_img = self.image_converter.convert(fz, 'force_z')
            vx_img = self.image_converter.convert(vx, 'vib_x')
            vy_img = self.image_converter.convert(vy, 'vib_y')
            vz_img = self.image_converter.convert(vz, 'vib_z')
            sound_img = self.image_converter.convert(sound, 'sound')

            # Apply image augmentation (if enabled)
            if self.transform and self.image_transforms is not None:
                # Generate random seed for consistency
                seed = torch.randint(0, 2 ** 32, (1,)).item()

                # Apply same random transformation to each modality
                torch.manual_seed(seed)
                fx_img = self.image_transforms(fx_img)

                torch.manual_seed(seed)
                fy_img = self.image_transforms(fy_img)

                torch.manual_seed(seed)
                fz_img = self.image_transforms(fz_img)

                torch.manual_seed(seed)
                vx_img = self.image_transforms(vx_img)

                torch.manual_seed(seed)
                vy_img = self.image_transforms(vy_img)

                torch.manual_seed(seed)
                vz_img = self.image_transforms(vz_img)

                torch.manual_seed(seed)
                sound_img = self.image_transforms(sound_img)

                # There's a 10% chance to randomly replace one modality with noise to enhance robustness
                if np.random.random() < 0.1 * self.aug_strength:
                    noise_idx = np.random.randint(0, 7)
                    noise_img = torch.randn_like(fx_img) * 0.1  # Low intensity noise
                    if noise_idx == 0:
                        fx_img = (fx_img * 0.8) + noise_img
                    elif noise_idx == 1:
                        fy_img = (fy_img * 0.8) + noise_img
                    elif noise_idx == 2:
                        fz_img = (fz_img * 0.8) + noise_img
                    elif noise_idx == 3:
                        vx_img = (vx_img * 0.8) + noise_img
                    elif noise_idx == 4:
                        vy_img = (vy_img * 0.8) + noise_img
                    elif noise_idx == 5:
                        vz_img = (vz_img * 0.8) + noise_img
                    else:
                        sound_img = (sound_img * 0.8) + noise_img

            return fx_img, fy_img, fz_img, vx_img, vy_img, vz_img, sound_img, torch.tensor(label, dtype=torch.long)

        else:
            fx = self.force_x[idx].copy()
            fy = self.force_y[idx].copy()
            fz = self.force_z[idx].copy()
            vx = self.vib_x[idx].copy()
            vy = self.vib_y[idx].copy()
            vz = self.vib_z[idx].copy()
            sound = self.sound[idx].copy()
            label = self.labels[idx]

            if self.transform:
                # Set different random seeds for different modalities to avoid correlated augmentations
                np.random.seed(int(idx * 1000 + np.random.randint(1000)))

                # Apply modality-specific augmentations
                fx = self._augment_force_signal(fx)
                fy = self._augment_force_signal(fy)
                fz = self._augment_force_signal(fz)

                vx = self._augment_vibration_signal(vx)
                vy = self._augment_vibration_signal(vy)
                vz = self._augment_vibration_signal(vz)

                sound = self._augment_sound_signal(sound)

            return (
                torch.from_numpy(fx).float(),
                torch.from_numpy(fy).float(),
                torch.from_numpy(fz).float(),
                torch.from_numpy(vx).float(),
                torch.from_numpy(vy).float(),
                torch.from_numpy(vz).float(),
                torch.from_numpy(sound).float(),
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


class MultiModalFusionThree(nn.Module):
    """Fusion module specially designed for three-modality data - force signal, vibration signal, sound signal"""

    def __init__(self, embed_dim, num_patches, num_heads, num_fusion_layers=3):
        super(MultiModalFusionThree, self).__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        # Modality-specific enhancement layers - automatically adjust parameters for different physical quantities
        self.modal_enhance = nn.ModuleDict({
            'force': nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.LayerNorm(embed_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
            ),
            'vib': nn.Sequential(
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
            )
        })

        # Modality internal attention
        self.modal_attention = nn.ModuleDict({
            'force': nn.MultiheadAttention(embed_dim, num_heads // 2, dropout=0.1),
            'vib': nn.MultiheadAttention(embed_dim, num_heads // 2, dropout=0.1),
            'sound': nn.MultiheadAttention(embed_dim, num_heads // 2, dropout=0.1)
        })

        # Fusion layers
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

        # Three-modality cross-attention (bidirectional)
        self.cross_modal_attention = nn.ModuleDict({
            'force_vib': nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1),
            'force_sound': nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1),
            'vib_force': nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1),
            'vib_sound': nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1),
            'sound_force': nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1),
            'sound_vib': nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)
        })

        # Modality normalization
        self.cross_norms = nn.ModuleDict({
            'force': nn.LayerNorm(embed_dim),
            'vib': nn.LayerNorm(embed_dim),
            'sound': nn.LayerNorm(embed_dim)
        })

        # Adaptive modality weight mechanism to handle differences in physical quantity units
        self.modality_weights = nn.Parameter(torch.ones(3) / 3)  # Initialize with equal weights

        # Gating fusion mechanism
        self.fusion_gates = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 3),
            nn.Sigmoid()
        )

        # Modality-specific linear layers
        self.modal_fc = nn.ModuleDict({
            'force': nn.Linear(embed_dim, embed_dim),
            'vib': nn.Linear(embed_dim, embed_dim),
            'sound': nn.Linear(embed_dim, embed_dim)
        })

    def forward(self, force_patches, vib_patches, sound_patches,
                force_weights, vib_weights, sound_weights):
        batch_size = force_patches.size(0)

        # Modality-specific transformations
        force_patches = self.modal_fc['force'](force_patches)
        vib_patches = self.modal_fc['vib'](vib_patches)
        sound_patches = self.modal_fc['sound'](sound_patches)

        # Feature enhancement
        force_enhanced = self.modal_enhance['force'](force_patches)
        vib_enhanced = self.modal_enhance['vib'](vib_patches)
        sound_enhanced = self.modal_enhance['sound'](sound_patches)

        # Apply modality-internal self-attention
        force_t = force_enhanced.transpose(0, 1)
        vib_t = vib_enhanced.transpose(0, 1)
        sound_t = sound_enhanced.transpose(0, 1)

        force_self, _ = self.modal_attention['force'](force_t, force_t, force_t)
        vib_self, _ = self.modal_attention['vib'](vib_t, vib_t, vib_t)
        sound_self, _ = self.modal_attention['sound'](sound_t, sound_t, sound_t)

        # Residual connections
        force_t = force_t + 0.2 * force_self
        vib_t = vib_t + 0.2 * vib_self
        sound_t = sound_t + 0.2 * sound_self

        # 添加这段代码来动态调整位置嵌入
        cls_token = self.cls_token.expand(batch_size, -1, -1)

        # 获取实际序列长度
        force_seq_len = force_patches.size(1) + 1  # +1 for cls token
        vib_seq_len = vib_patches.size(1) + 1
        sound_seq_len = sound_patches.size(1) + 1

        # 调整位置嵌入以匹配序列长度
        force_pos_embed = self._adjust_positional_encoding(self.pos_embed, force_seq_len)
        vib_pos_embed = self._adjust_positional_encoding(self.pos_embed, vib_seq_len)
        sound_pos_embed = self._adjust_positional_encoding(self.pos_embed, sound_seq_len)

        # Add CLS token and position encoding with adjusted positional embeddings
        force_input = torch.cat([cls_token, force_patches], dim=1) + force_pos_embed
        vib_input = torch.cat([cls_token, vib_patches], dim=1) + vib_pos_embed
        sound_input = torch.cat([cls_token, sound_patches], dim=1) + sound_pos_embed

        # Apply dynamic weights
        force_input = force_input * (1.0 + force_weights.unsqueeze(1))
        vib_input = vib_input * (1.0 + vib_weights.unsqueeze(1))
        sound_input = sound_input * (1.0 + sound_weights.unsqueeze(1))

        # Transpose to required format for attention layers
        force_t = force_input.transpose(0, 1)
        vib_t = vib_input.transpose(0, 1)
        sound_t = sound_input.transpose(0, 1)

        # Apply adaptive modality weights to balance contributions from different physical quantities
        normalized_weights = F.softmax(self.modality_weights, dim=0)
        force_t = force_t * normalized_weights[0]
        vib_t = vib_t * normalized_weights[1]
        sound_t = sound_t * normalized_weights[2]

        # Bidirectional inter-modality attention calculation
        force_vib, _ = self.cross_modal_attention['force_vib'](force_t, vib_t, vib_t)
        force_sound, _ = self.cross_modal_attention['force_sound'](force_t, sound_t, sound_t)
        vib_force, _ = self.cross_modal_attention['vib_force'](vib_t, force_t, force_t)
        vib_sound, _ = self.cross_modal_attention['vib_sound'](vib_t, sound_t, sound_t)
        sound_force, _ = self.cross_modal_attention['sound_force'](sound_t, force_t, force_t)
        sound_vib, _ = self.cross_modal_attention['sound_vib'](sound_t, vib_t, vib_t)

        # Apply residual connections and layer normalization
        force_t = self.cross_norms['force'](force_t + force_vib + force_sound)
        vib_t = self.cross_norms['vib'](vib_t + vib_force + vib_sound)
        sound_t = self.cross_norms['sound'](sound_t + sound_force + sound_vib)

        # Implement dynamic gated fusion
        combined = torch.cat([
            force_t.transpose(0, 1).mean(dim=1),
            vib_t.transpose(0, 1).mean(dim=1),
            sound_t.transpose(0, 1).mean(dim=1)
        ], dim=1)

        gate_outputs = self.fusion_gates(combined)
        gate_chunks = gate_outputs.chunk(3, dim=1)
        force_gate, vib_gate, sound_gate = gate_chunks

        # Apply gating mechanism
        force_gate = force_gate.unsqueeze(0)
        vib_gate = vib_gate.unsqueeze(0)
        sound_gate = sound_gate.unsqueeze(0)
        force_t = force_t * force_gate
        vib_t = vib_t * vib_gate
        sound_t = sound_t * sound_gate

        # Fuse and apply multi-layer transformation
        fused = torch.cat([force_t, vib_t, sound_t], dim=0)
        residual_fused = fused

        for layer in self.fusion_layers:
            # Multi-head attention
            attn_out, _ = layer['attn'](fused, fused, fused)
            fused = layer['norm1'](fused + attn_out)

            # Feedforward network
            ffn_out = layer['ffn'](fused)
            fused = layer['norm2'](fused + ffn_out)

            # Add scaled residual connection
            fused = fused + residual_fused * 0.1

        # Return batch-priority format
        return fused.transpose(0, 1)

    def _adjust_positional_encoding(self, pos_embed, target_len):
        """动态调整位置嵌入大小以匹配目标长度"""
        # 获取当前位置嵌入的尺寸
        current_len = pos_embed.shape[1]

        # 如果目标长度与当前长度相同，则无需调整
        if current_len == target_len:
            return pos_embed

        # 分离类别标记嵌入和其他位置嵌入
        cls_pos_embed = pos_embed[:, 0:1, :]
        other_pos_embed = pos_embed[:, 1:, :]

        # 根据目标长度进行插值
        if target_len > 1:
            # 调整其他位置嵌入的大小（除了cls token的位置）
            other_pos_embed = F.interpolate(
                other_pos_embed.permute(0, 2, 1),
                size=target_len - 1,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)

            # 合并回类别标记嵌入
            adjusted_pos_embed = torch.cat([cls_pos_embed, other_pos_embed], dim=1)

            #print(f"位置嵌入已调整: {current_len} -> {target_len}")
            return adjusted_pos_embed
        else:
            # 如果目标长度小于等于1，则只返回类别标记嵌入
            return cls_pos_embed


class TransformerWithPDF(nn.Module):
    def __init__(self, input_shape, num_heads, num_patches, projection_dim, num_classes, image_input=False,
                 dropout_rate=0.3):
        super(TransformerWithPDF, self).__init__()
        self.input_shape = input_shape  # [force_x, force_y, force_z, vib_x, vib_y, vib_z, sound] length
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.image_input = image_input
        self.dropout_rate = dropout_rate

        if image_input:
            # Create CNN encoders for each input modality
            self.force_x_encoder = self._build_enhanced_cnn_encoder()
            self.force_y_encoder = self._build_enhanced_cnn_encoder()
            self.force_z_encoder = self._build_enhanced_cnn_encoder()
            self.vib_x_encoder = self._build_enhanced_cnn_encoder()
            self.vib_y_encoder = self._build_enhanced_cnn_encoder()
            self.vib_z_encoder = self._build_enhanced_cnn_encoder()
            self.sound_encoder = self._build_enhanced_cnn_encoder()
        else:
            # Create independent encoders for each modality
            self.force_x_encoder = self._build_modal_encoder(input_shape[0])
            self.force_y_encoder = self._build_modal_encoder(input_shape[1])
            self.force_z_encoder = self._build_modal_encoder(input_shape[2])
            self.vib_x_encoder = self._build_modal_encoder(input_shape[3])
            self.vib_y_encoder = self._build_modal_encoder(input_shape[4])
            self.vib_z_encoder = self._build_modal_encoder(input_shape[5])
            self.sound_encoder = self._build_modal_encoder(input_shape[6])

        # Force signal fusion transformer (first fuse three-axis force)
        self.force_transformer = nn.ModuleList([
            self._build_transformer_layer() for _ in range(2)
        ])

        # Vibration signal fusion transformer (first fuse three-axis vibration)
        self.vib_transformer = nn.ModuleList([
            self._build_transformer_layer() for _ in range(2)
        ])

        # Sound signal processing layers
        self.sound_transformer = nn.ModuleList([
            self._build_transformer_layer() for _ in range(2)
        ])

        # Add specific modality normalization layers to handle different physical quantity units
        self.force_norm = nn.LayerNorm(projection_dim)
        self.vibration_norm = nn.LayerNorm(projection_dim)
        self.sound_norm = nn.LayerNorm(projection_dim)

        # Modify fusion layer to three-way fusion (force, vibration, sound)
        self.fusion_layer = MultiModalFusionThree(
            embed_dim=self.projection_dim,
            num_patches=self.num_patches,
            num_heads=self.num_heads,
            num_fusion_layers=3
        )

        # Prediction modules for each modality
        self.force_prediction_module = PredictionModule(input_dim=projection_dim, output_dim=1)
        self.vib_prediction_module = PredictionModule(input_dim=projection_dim, output_dim=1)
        self.sound_prediction_module = PredictionModule(input_dim=projection_dim, output_dim=1)

        # Main transformer layers
        self.transformer_layers = nn.ModuleList([
            self._build_transformer_layer() for _ in range(6)
        ])

        # Classifiers
        self.mid_classifier = self._build_enhanced_classifier(num_classes)
        self.force_classifier = self._build_enhanced_classifier(num_classes)
        self.vib_classifier = self._build_enhanced_classifier(num_classes)
        self.sound_classifier = self._build_enhanced_classifier(num_classes)
        self.fc_out = self._build_enhanced_output_classifier(num_classes)

        # Weight initialization
        self._initialize_weights()

    def _build_modal_encoder(self, input_dim):
        """Create encoder for each modality"""
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
            nn.Dropout(self.dropout_rate * 0.7),  # Lower dropout for classifiers
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

    def _build_enhanced_cnn_encoder(self):
        """Build enhanced CNN encoder for image inputs"""
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

    def _build_transformer_layer(self):
        return nn.TransformerEncoderLayer(
            d_model=self.projection_dim,
            nhead=self.num_heads,
            dim_feedforward=self.projection_dim * 4,
            dropout=0.2,
            activation='gelu',
            batch_first=True
        )

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use kaiming_normal to initialize linear layers
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

    def forward(self, force_x, force_y, force_z, vib_x, vib_y, vib_z, sound):
        # Encode 7 input signals
        if self.image_input:
            fx_patches = self.force_x_encoder(force_x)
            fy_patches = self.force_y_encoder(force_y)
            fz_patches = self.force_z_encoder(force_z)
            vx_patches = self.vib_x_encoder(vib_x)
            vy_patches = self.vib_y_encoder(vib_y)
            vz_patches = self.vib_z_encoder(vib_z)
            sound_patches = self.sound_encoder(sound)
        else:
            fx_patches = self.force_x_encoder(force_x)
            fy_patches = self.force_y_encoder(force_y)
            fz_patches = self.force_z_encoder(force_z)
            vx_patches = self.vib_x_encoder(vib_x)
            vy_patches = self.vib_y_encoder(vib_y)
            vz_patches = self.vib_z_encoder(vib_z)
            sound_patches = self.sound_encoder(sound)

        # Save original patches for residual connections
        fx_original = fx_patches
        fy_original = fy_patches
        fz_original = fz_patches
        vx_original = vx_patches
        vy_original = vy_patches
        vz_original = vz_patches
        sound_original = sound_patches

        # 1. First fuse three-axis force signals - simple concatenation and attention mechanism
        # Concatenate three-axis force signals
        force_combined = torch.cat([fx_patches, fy_patches, fz_patches], dim=1)  # Concatenate along sequence dimension

        # Apply force signal dedicated transformer
        force_patches = force_combined
        for layer in self.force_transformer:
            force_patches = layer(force_patches)
            # Add residual connection
            force_patches = force_patches + 0.1 * torch.cat([fx_original, fy_original, fz_original], dim=1)

        # Apply specific normalization to force signal
        force_patches = self.force_norm(force_patches)  # Specific normalization layer

        # 2. Fuse three-axis vibration signals
        # Concatenate three-axis vibration signals
        vib_combined = torch.cat([vx_patches, vy_patches, vz_patches], dim=1)

        # Apply vibration signal dedicated transformer
        vib_patches = vib_combined
        for layer in self.vib_transformer:
            vib_patches = layer(vib_patches)
            # Add residual connection
            vib_patches = vib_patches + 0.1 * torch.cat([vx_original, vy_original, vz_original], dim=1)

        # Apply specific normalization to vibration signal
        vib_patches = self.vibration_norm(vib_patches)

        # 3. Process sound signal
        sound_processed = sound_patches
        for layer in self.sound_transformer:
            sound_processed = layer(sound_processed)
            sound_processed = sound_processed + 0.1 * sound_original

        # Apply specific normalization to sound signal
        sound_processed = self.sound_norm(sound_processed)

        # Extract features from each modality
        force_feat = force_patches.mean(dim=1)  # Average pooling
        vib_feat = vib_patches.mean(dim=1)
        sound_feat = sound_processed.mean(dim=1)

        # Modality-specific prediction
        force_pred = self.force_classifier(force_feat)
        vib_pred = self.vib_classifier(vib_feat)
        sound_pred = self.sound_classifier(sound_feat)

        # Calculate modality weights
        force_weights = self.force_prediction_module(force_feat)
        vib_weights = self.vib_prediction_module(vib_feat)
        sound_weights = self.sound_prediction_module(sound_feat)

        # Apply fusion layer
        fused_features = self.fusion_layer(force_patches, vib_patches, sound_processed,
                                           force_weights, vib_weights, sound_weights)

        # Extract CLS token for classification (first token)
        cls_token = fused_features[:, 0]

        # Process with main transformer layers
        main_features = fused_features
        for layer in self.transformer_layers:
            main_features = layer(main_features)

        # Extract final features for classification
        final_feat = main_features[:, 0]  # Use CLS token

        # Final classification
        logits = self.fc_out(final_feat)

        if self.training:
            # During training, return all predictions for multi-task learning
            return {
                'main': logits,
                'force': force_pred,
                'vibration': vib_pred,
                'sound': sound_pred,
                'mid': self.mid_classifier(cls_token)
            }
        else:
            # During inference, just return main prediction
            return logits


# 1. Enhanced Mixed Loss
class EnhancedMixedLoss(nn.Module):
    """混合损失函数，结合了交叉熵、Focal Loss和标签平滑"""

    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1, temp=1.0):
        super(EnhancedMixedLoss, self).__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # Focal Loss参数
        self.smoothing = smoothing  # 标签平滑参数
        self.temp = temp  # 温度缩放参数

    def forward(self, pred, target):
        # 定义一个变量来存储原始字典（如果有的话）
        pred_dict = None

        # 首先处理预测输出，确保我们有一个有效的预测张量
        if isinstance(pred, dict):
            # 提取预测张量
            modal_weights = {
                "main": 0.6,
                "force": 0.15,
                "vibration": 0.15,
                "sound": 0.1
            }

            # 保存原始字典以便后续使用
            pred_dict = pred

            # 使用加权平均组合所有模态的预测
            combined_pred = None
            weight_sum = 0

            for modality, weight in modal_weights.items():
                if modality in pred_dict and isinstance(pred_dict[modality], torch.Tensor):
                    modal_tensor = pred_dict[modality] / self.temp  # 应用温度缩放
                    if combined_pred is None:
                        combined_pred = weight * modal_tensor
                    else:
                        combined_pred += weight * modal_tensor
                    weight_sum += weight

            # 确保权重总和不为零
            if weight_sum > 0 and combined_pred is not None:
                combined_pred = combined_pred / weight_sum

            # 如果没有找到有效的预测张量，抛出错误
            if combined_pred is None:
                available_keys = [k for k in pred_dict.keys() if isinstance(pred_dict[k], torch.Tensor)]
                raise KeyError(f"在模型输出中未找到任何可用的预测张量。可用的键: {available_keys}")

            # 使用组合预测作为主预测
            pred = combined_pred
        else:
            # 如果输入已经是张量，直接应用温度缩放
            pred = pred / self.temp

        # 现在 pred 是一个张量，可以进行正常的损失计算
        # 计算log softmax
        log_softmax = F.log_softmax(pred, dim=-1)

        # 获取类别数量
        n_classes = pred.size(1)

        # 创建平滑的one-hot目标
        target_one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        target_smooth = (1 - self.smoothing) * target_one_hot + self.smoothing / n_classes

        # 计算基本损失
        loss = -target_smooth * log_softmax

        # 应用Focal Loss权重
        if self.gamma > 0:
            probs = torch.exp(log_softmax)
            pt = torch.gather(probs, 1, target.unsqueeze(1))
            focal_weight = (1 - pt) ** self.gamma
            loss = focal_weight * loss

        # 应用类别权重
        if self.alpha is not None:
            alpha_weight = torch.gather(self.alpha.expand(pred.size(0), -1), 1, target.unsqueeze(1))
            loss = alpha_weight * loss

        # 计算多模态损失（如果可用）
        multi_modal_loss = 0.0
        if pred_dict is not None:  # 使用 pred_dict 是否为 None 进行检查，而不是检查它是否是字典
            # 对每个模态分别计算损失
            modal_loss_weights = {
                "main": 0.6,  # 主要预测的损失权重较高
                "force": 0.15,
                "vibration": 0.15,
                "sound": 0.1
            }

            for modality, weight in modal_loss_weights.items():
                if modality in pred_dict and isinstance(pred_dict[modality], torch.Tensor):
                    # 跳过已经作为主预测使用的张量
                    modal_tensor = pred_dict[modality] / self.temp
                    modal_log_softmax = F.log_softmax(modal_tensor, dim=-1)
                    modal_target_one_hot = torch.zeros_like(modal_tensor).scatter_(1, target.unsqueeze(1), 1)
                    modal_target_smooth = (
                                                      1 - self.smoothing) * modal_target_one_hot + self.smoothing / modal_tensor.size(
                        1)
                    modal_loss = (-modal_target_smooth * modal_log_softmax).sum(dim=1).mean()
                    multi_modal_loss += weight * modal_loss

        # 返回总损失 (主损失 + 多模态损失)
        total_loss = loss.sum(dim=1).mean() + 0.5 * multi_modal_loss
        return total_loss


# 2. Learning Rate Scheduler
def get_enhanced_lr_scheduler(optimizer, args, train_loader=None):
    """创建增强型学习率调度器"""

    if args.lr_scheduler == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    elif args.lr_scheduler == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    elif args.lr_scheduler == 'cosine_warm':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    elif args.lr_scheduler == 'one_cycle':
        if train_loader is None:
            raise ValueError("train_loader is required for OneCycleLR")
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr * 10,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader)
        )

    elif args.lr_scheduler == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

    elif args.lr_scheduler == 'cosine_warmup':
        # 自定义的余弦退火+预热
        warmup_epochs = int(args.epochs * 0.1)  # 预热时间为总训练时间的10%

        def lambda_lr(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    elif args.lr_scheduler == 'cyclic':
        return optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=args.lr / 10,
            max_lr=args.lr,
            step_size_up=len(train_loader) * 5,
            mode='triangular2'
        )

    else:
        raise ValueError(f"Unsupported scheduler: {args.lr_scheduler}")


# 3. Balance multimodal dataset
def balance_multimodal_dataset(force_x, force_y, force_z, vib_x, vib_y, vib_z, sound, labels, sampling_strategy='auto'):
    """使用SMOTE平衡多模态数据集"""
    # 合并特征以便SMOTE处理
    n_samples = force_x.shape[0]

    # 展平每个模态
    force_x_flat = force_x.reshape(n_samples, -1)
    force_y_flat = force_y.reshape(n_samples, -1)
    force_z_flat = force_z.reshape(n_samples, -1)
    vib_x_flat = vib_x.reshape(n_samples, -1)
    vib_y_flat = vib_y.reshape(n_samples, -1)
    vib_z_flat = vib_z.reshape(n_samples, -1)
    sound_flat = sound.reshape(n_samples, -1)

    # 将所有特征连接起来
    X = np.concatenate((
        force_x_flat, force_y_flat, force_z_flat,
        vib_x_flat, vib_y_flat, vib_z_flat,
        sound_flat
    ), axis=1)

    # 应用SMOTE过采样
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, labels)

    # 计算每个模态的特征维度
    dims = [
        force_x_flat.shape[1],
        force_y_flat.shape[1],
        force_z_flat.shape[1],
        vib_x_flat.shape[1],
        vib_y_flat.shape[1],
        vib_z_flat.shape[1],
        sound_flat.shape[1]
    ]

    # 截取点
    cuts = np.cumsum([0] + dims)

    # 拆分回原来的模态形状
    force_x_res = X_resampled[:, cuts[0]:cuts[1]].reshape(-1, force_x.shape[1])
    force_y_res = X_resampled[:, cuts[1]:cuts[2]].reshape(-1, force_y.shape[1])
    force_z_res = X_resampled[:, cuts[2]:cuts[3]].reshape(-1, force_z.shape[1])
    vib_x_res = X_resampled[:, cuts[3]:cuts[4]].reshape(-1, vib_x.shape[1])
    vib_y_res = X_resampled[:, cuts[4]:cuts[5]].reshape(-1, vib_y.shape[1])
    vib_z_res = X_resampled[:, cuts[5]:cuts[6]].reshape(-1, vib_z.shape[1])
    sound_res = X_resampled[:, cuts[6]:].reshape(-1, sound.shape[1])

    print(f"原始样本分布: {np.bincount(labels)}")
    print(f"重采样后样本分布: {np.bincount(y_resampled)}")

    return force_x_res, force_y_res, force_z_res, vib_x_res, vib_y_res, vib_z_res, sound_res, y_resampled


# 4. Plot confusion matrix
def plot_confusion_matrix(cm, class_names, save_path=None):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# 5. Plot cross-validation summary
def plot_cross_validation_summary(cv_results, save_dir):
    """绘制交叉验证结果汇总"""
    # 绘制F1分数
    plt.figure(figsize=(10, 6))
    folds = list(range(1, len(cv_results['fold_f1_scores']) + 1))
    plt.bar(folds, cv_results['fold_f1_scores'], color='skyblue', alpha=0.7)
    plt.axhline(y=cv_results['mean_f1'], color='r', linestyle='-', label=f"Mean F1: {cv_results['mean_f1']:.4f}")

    # 添加标准差范围
    plt.fill_between(
        [0.5, len(folds) + 0.5],
        [cv_results['mean_f1'] - cv_results['std_f1']] * 2,
        [cv_results['mean_f1'] + cv_results['std_f1']] * 2,
        alpha=0.2, color='red'
    )

    plt.xlabel('Fold')
    plt.ylabel('F1 Score')
    plt.title('Cross-Validation F1 Scores')
    plt.xticks(folds)
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(save_dir, 'cv_f1_scores.png'), bbox_inches='tight')
    plt.close()

    # 如果有AUC分数，也绘制
    if 'fold_auc_scores' in cv_results and cv_results['fold_auc_scores'][0] > 0:
        plt.figure(figsize=(10, 6))
        plt.bar(folds, cv_results['fold_auc_scores'], color='lightgreen', alpha=0.7)
        plt.axhline(y=cv_results['mean_auc'], color='r', linestyle='-', label=f"Mean AUC: {cv_results['mean_auc']:.4f}")

        # 添加标准差范围
        plt.fill_between(
            [0.5, len(folds) + 0.5],
            [cv_results['mean_auc'] - cv_results['std_auc']] * 2,
            [cv_results['mean_auc'] + cv_results['std_auc']] * 2,
            alpha=0.2, color='red'
        )

        plt.xlabel('Fold')
        plt.ylabel('AUC Score')
        plt.title('Cross-Validation AUC Scores')
        plt.xticks(folds)
        plt.ylim([0, 1])
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(save_dir, 'cv_auc_scores.png'), bbox_inches='tight')
        plt.close()