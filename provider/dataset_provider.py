import os
import random
import numpy as np

import rasterio as rio

import torch

from torch.utils.data import Dataset, DataLoader

import torchvision
torchvision.disable_beta_transforms_warning()

from torchvision.transforms import v2


transforms = torchvision.transforms.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip()
])


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


class NrwDataSet(Dataset):
    def __init__(self, data_dir, load_amount):

        self.data_dir = data_dir
        self.dataset = []

        frame_files = os.listdir(data_dir)

        c = 0
        for frame_file in frame_files:
            frame = rio.open(os.path.join(data_dir, frame_file), allow_pickle=True)

            self.dataset.append(frame)

            c += 1

            if load_amount > 0:
                if c == load_amount:
                    break

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        data_tuple = self.dataset[index]

        data = torch.Tensor(data_tuple[0])
        mask = torch.Tensor(data_tuple[1].astype(np.uint8))

        frame_file = data_tuple[3]

        return data, mask, frame_file


def get_loader(npz_dir, batch_size, num_workers=2, pin_memory=True, shuffle=True, load_amount=0):
    train_ds = NrwDataSet(npz_dir, load_amount)
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


# Debug method if we want to get directly the data slices, not the whole loader
def get_dataset(npz_dir, load_amount=0):
    return NrwDataSet(npz_dir, load_amount)