import h5py
import random
import torch
import numpy as np
import pickle

from utils.data.transforms import FastmriDataTransform, CustomMaskFunc
from utils.data.data_augment import DataAugmentor

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from functools import partial
from typing import Callable, Optional

def worker_init_fn(worker_id, seed):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)

def calculate_mask_acc(mask):
    mask = mask.copy()
    cent = mask.shape[0] // 2

    left = np.argmin(np.flip(mask[:cent]))
    right = np.argmin(mask[cent:])
    num_low_freqs = left + right

    pad = (mask.shape[0] - num_low_freqs + 1) // 2
    mask[pad:pad + num_low_freqs] = 0

    acc = (mask.shape[0] - num_low_freqs) / np.sum(mask)

    return 'acc4' if abs(acc - 4) < abs(acc - 8) else 'acc8'

class TestSliceData(Dataset):
    def __init__(self, root: Path):
        self.image_examples = []
        self.kspace_examples = []

        image_files = list(Path(root/"image").iterdir())
        for fname in sorted(image_files):
            self.image_examples += [fname]
        kspace_files = list(Path(root/"kspace").iterdir())
        for fname in sorted(kspace_files):
            self.kspace_examples += [fname]
        
    def __len__(self):
        return len(self.image_examples)
    
    def __getitem__(self, i):
        image_fname = self.image_examples[i]
        kspace_fname = self.kspace_examples[i]
        if image_fname.name != kspace_fname.name:
            raise ValueError(f"Image file {image_fname.name} does not match kspace file {kspace_fname.name}")
        
        with h5py.File(image_fname, "r") as hf:
            image = np.array(hf['image_grappa'])
            image = torch.from_numpy(image.astype(np.float32))
        with h5py.File(kspace_fname, "r") as hf:
            kspace = np.array(hf['kspace'])
            mask = np.array(hf['mask'])
            acc = calculate_mask_acc(mask)

            kspace = torch.from_numpy(np.stack((kspace.real, kspace.imag), axis=-1).astype(np.float32))
            mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).to(torch.uint8)

        return mask, kspace, image, acc, kspace_fname.name

class CNNSliceData(Dataset):
    def __init__(self, root: Path):
        self.image_examples = []

        for task in ['brain', 'knee']:
            image_files = list(Path(root/task/"image").iterdir())
            for fname in sorted(image_files):
                self.image_examples += [fname]

    def __len__(self):
        return len(self.image_examples)
    
    def __getitem__(self, i):
        image_fname = self.image_examples[i]
        with h5py.File(image_fname, "r") as hf:
            image = np.array(hf['image_grappa'])
        if 'brain' in image_fname.name:
            target = 0
        elif 'knee' in image_fname.name:
            target = 1
        return torch.from_numpy(image.astype(np.float32)), torch.tensor(target, dtype=torch.long)

class FastmriSliceData(Dataset):
    def __init__(
        self,
        root: Path,
        transform: Optional[Callable] = None,
        use_dataset_cache: bool = False,
        volume_sample_rate: Optional[float] = None,
        image_cache_file: Path = Path("image_cache.pkl"),
        kspace_cache_file: Path = Path("kspace_cache.pkl"),
        num_adj_slices: int = 1,
        input_key: str = "kspace",
        target_key: str = "image_label",
    ):
        
        assert num_adj_slices % 2 == 1, "Number of adjacent slices must be odd in SliceDataset"
        self.num_adj_slices = num_adj_slices
        self.start_adj, self.end_adj = -(self.num_adj_slices//2), self.num_adj_slices//2+1

        self.image_cache_file = root / image_cache_file
        self.kspace_cache_file = root / kspace_cache_file
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key

        self.image_examples = []
        self.kspace_examples = []

        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        if self.image_cache_file.exists() and use_dataset_cache:
            with open(self.image_cache_file, "rb") as f:
                image_cache = pickle.load(f)
        else:
            image_cache = {}
        
        if self.kspace_cache_file.exists() and use_dataset_cache:
            with open(self.kspace_cache_file, "rb") as f:
                kspace_cache = pickle.load(f)
        else:
            kspace_cache = {}

        if image_cache.get(root) is None or not use_dataset_cache:
            image_files = list(Path(root / "image").iterdir())
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)

                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]
            if image_cache.get(root) is None and use_dataset_cache:
                image_cache[root] = self.image_examples
                with open(self.image_cache_file, "wb") as cache_f:
                    pickle.dump(image_cache, cache_f)
        else:
            self.image_examples = image_cache[root]
        
        if kspace_cache.get(root) is None or not use_dataset_cache:
            kspace_files = list(Path(root / "kspace").iterdir())
            for fname in sorted(kspace_files):
                num_slices = self._get_metadata(fname)

                self.kspace_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]
            if kspace_cache.get(root) is None and use_dataset_cache:
                kspace_cache[root] = self.kspace_examples
                with open(self.kspace_cache_file, "wb") as cache_f:
                    pickle.dump(kspace_cache, cache_f)
        else:
            self.kspace_examples = kspace_cache[root]

        if volume_sample_rate < 1.0:
            vol_names = sorted(list(set([f[0].stem for f in self.kspace_examples])))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.kspace_examples = [
                kspace_example
                for kspace_example in self.kspace_examples
                if kspace_example[0].stem in sampled_vols
            ]
            self.image_examples = [
                image_example
                for image_example in self.image_examples
                if image_example[0].stem in sampled_vols
            ]

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices
    
    def __len__(self):
        return len(self.kspace_examples)
    
    def _get_frames_indices(self, data_slice, num_slices):
        z_list = [min(max(i+data_slice, 0), num_slices-1)
                  for i in range(self.start_adj, self.end_adj)]
        return z_list
    
    def __getitem__(self, i):
        image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]
        if image_fname.name != kspace_fname.name:
            raise ValueError(f"Image file {image_fname.name} does not match kspace file {kspace_fname.name}")

        input = []
        with h5py.File(kspace_fname, "r") as hf:
            num_slices = hf[self.input_key].shape[0]
            slice_idx_list = self._get_frames_indices(dataslice, num_slices)
            for slice_idx in slice_idx_list:
                input.append(hf["kspace"][slice_idx])
            input = np.concatenate(input, axis=0)
            mask =  np.array(hf["mask"])
            
        with h5py.File(image_fname, "r") as hf:
            target = hf[self.target_key][dataslice]
            attrs = dict(hf.attrs)
        
        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice)


def create_data_loaders(data_path, args, shuffle=False, data_type='train', slicedata='FastmriSliceData'):
    if slicedata == 'FastmriSliceData':
        data_storage = FastmriSliceData(
            root=data_path/args.task,
            transform=FastmriDataTransform(
                task=args.task,
                max_key=args.max_key,
                num_adj_slices=args.num_adj_slices if hasattr(args, 'num_adj_slices') else 1,
                mask_func=CustomMaskFunc(args.acceleration, args.mask_type, args.seed),
                augmentor=DataAugmentor(hparams=args)
            ),
            use_dataset_cache=(args.volume_sample_rate==1.0),
            volume_sample_rate=args.volume_sample_rate,
            num_adj_slices=args.num_adj_slices if hasattr(args, 'num_adj_slices') else 1,
            input_key=args.input_key,
            target_key=args.target_key,
        )
    elif slicedata == 'CNNSliceData':
        data_storage = CNNSliceData(root=data_path)
    elif slicedata == 'TestSliceData':
        data_storage = TestSliceData(root=data_path)

    worker_init = partial(worker_init_fn, seed=args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=1,
        shuffle=shuffle,
        generator=g,
        num_workers=3 if data_type=='train' else 1,
        persistent_workers=False,
        pin_memory=True,
        worker_init_fn=worker_init,
    )
    return data_loader
