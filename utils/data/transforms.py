import numpy as np
import torch
from utils.model.fastmri import fft2c, ifft2c, rss_complex
from typing import Sequence

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data.astype(np.float32))

class CustomMaskFunc():
    def __init__(self, acceleration: int, mask_type: str='fixed', seed: int=430):
        self.acceleration = acceleration
        self.mask_type = mask_type
        if mask_type != 'fixed':
            self.rng = np.random.RandomState()
            self.rng.seed(seed)

    def __call__(self, shape: Sequence[int]) -> torch.Tensor:
        num_cols = shape[-2]
        num_low_freqs = int(round(num_cols * 0.08))
        mask = np.zeros(num_cols, dtype=np.float32)

        if self.mask_type == 'random_spaced':
            num_bins = num_cols // self.acceleration
            for i in range(num_bins):
                start = i * self.acceleration
                end = min(start + self.acceleration, num_cols)
                if  start < end:
                    idx = self.rng.randint(start, end)
                    mask[idx] = True
        else:
            if self.mask_type == 'random_offset':
                offset = self.rng.randint(0, self.acceleration)
            else:
                offset = (num_cols // 2) % self.acceleration
            accel_samples = np.arange(offset, num_cols - 1, self.acceleration)
            accel_samples = np.around(accel_samples).astype(np.uint)
            mask[accel_samples] = True

        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = True
        mask = to_tensor(mask.reshape(1, 1, num_cols, 1))

        return mask

class FastmriDataTransform:
    def __init__(
        self, 
        task: str = 'brain',
        max_key: str = 'max',
        num_adj_slices: int = 1,
        mask_func = None,
        augmentor = None,
    ):
        self.max_key = max_key
        self.num_adj_slices = num_adj_slices
        if task == 'knee':
            self.uniform_height = 416
        else:
            self.uniform_height = 384
        self.mask_func = mask_func
        self.augmentor = augmentor

    def center_crop(self, data, height, width):
        _, h, w, _ = data.shape

        if h < height:
            pad_h1 = (height - h) // 2
            pad_h2 = (height - h) - pad_h1
            data = torch.nn.functional.pad(data.permute(0, 3, 1, 2), (0, 0, pad_h1, pad_h2), mode='constant', value=0).permute(0, 2, 3, 1)
            h = height

        if w < width:
            pad_w1 = (width - w) // 2
            pad_w2 = (width - w) - pad_w1
            data = torch.nn.functional.pad(data.permute(0, 3, 1, 2), (pad_w1, pad_w2, 0, 0), mode='constant', value=0).permute(0, 2, 3, 1)
            w = width

        start_h = (h - height) // 2
        start_w = (w - width) // 2
        return data[:, start_h:start_h + height, start_w:start_w + width, :]
    
    def __call__(self, mask, input, target, attrs, fname, slice_idx):
        target = to_tensor(target)
        maximum = attrs[self.max_key]

        kspace = to_tensor(input)
        image = ifft2c(kspace)
        image, is_aug = self.augmentor(image)
        if is_aug:
            target = torch.chunk(image, self.num_adj_slices)[self.num_adj_slices // 2]
            target = rss_complex(self.center_crop(target, 384, 384))
            maximum = target.max().item()

        if self.uniform_height < image.shape[-3]:
            h_from = (image.shape[-3] - self.uniform_height) // 2
            h_to = h_from + self.uniform_height
            image = image[..., h_from:h_to, :, :]
        kspace = fft2c(image)
        mask = self.mask_func(kspace.shape)
        kspace = kspace * mask + 0.0
    
        mask = mask.to(torch.uint8)
        return mask, kspace, target, maximum, fname, slice_idx