import os
import random
import numpy as np

import rasterio as rio

import torch

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torchvision
torchvision.disable_beta_transforms_warning()

from torchvision.transforms import v2


transforms = torchvision.transforms.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip()
])


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


class NrwDataSet(Dataset):
    def __init__(self, data_dir, augmentation, load_amount):

        self.data_dir = data_dir
        self.augmentation = augmentation
        self.dataset = []

        frame_files = os.listdir(data_dir)

        c = 0
        for frame_file in tqdm(frame_files):
            with rio.open(os.path.join(data_dir, frame_file)) as src:
                self.dataset.append((np.expand_dims(src.read(1), axis=0), src.read((range(2, 22))), frame_file))

                src.close()

            c += 1

            if load_amount > 0:
                if c == load_amount:
                    break

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        data_tuple = self.dataset[index]

        data = torch.Tensor(data_tuple[1])
        mask = torch.Tensor(data_tuple[0])

        if self.augmentation:
            data = transforms(data)
            mask = transforms(mask)

        frame_file = data_tuple[2]

        return data, mask, frame_file


def get_loader(npz_dir, augmentation, batch_size, num_workers=2, pin_memory=True, shuffle=True, load_amount=0):
    train_ds = NrwDataSet(npz_dir, augmentation, load_amount)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=seed_worker,
        generator=g
    )
    return train_loader


def get_loader_from_dataset(dataset: NrwDataSet, batch_size, num_workers=0, pin_memory=False, shuffle=True):
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_skip_none
    )
    return train_loader


def get_dataset(npz_dir, augmentation, load_amount=0):
    return NrwDataSet(npz_dir, augmentation, load_amount)