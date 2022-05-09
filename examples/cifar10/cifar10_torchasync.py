import sys

import torch

sys.path.insert(0, '../..')

import multiprocessing as mp
import pickle

import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.models.vgg import cfgs, make_layers

from torch_async import Model, ChunkDataloader
from torchvision.models import VGG
from os.path import join


class VGG11Async(VGG, Model):
    def __init__(self, **kwargs):
        cfg = 'A'
        batch_norm = True
        super().__init__(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)

    def forward(self, x):
        x = x.reshape(-1, 3, 32, 32)

        x = super().forward(x)

        return x


class CIFAR10ChunkDataloader(ChunkDataloader):
    def __init__(self, train):
        root_folder = './data'
        dataset = CIFAR10(root=root_folder, download=True)

        chunk_list = dataset.train_list if train else dataset.test_list

        self.chunk_paths = [join(root_folder, dataset.base_folder, filename) for filename, _ in chunk_list]

        self.num_chunks_ = len(self.chunk_paths)
        self.num_items_ = [self.load_chunk(i)[0] for i in range(self.num_chunks_)]
        self.sample_size_ = 3 * 32 * 32
        self.num_samples_ = sum(self.num_items_)

    def load_chunk(self, index):
        file_path = self.chunk_paths[index]

        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            data = entry['data']
            if 'labels' in entry:
                targets = entry['labels']
            else:
                targets = entry['fine_labels']

            targets = np.array(targets)

        return len(data), data, targets

    def sample_size(self):
        return self.sample_size_

    def num_items(self):
        return self.num_items_

    def num_chunks(self):
        return self.num_chunks_

    def __len__(self):
        return self.num_samples_


if __name__ == '__main__':
    mp.set_start_method('spawn')

    train_dataloader = CIFAR10ChunkDataloader(train=True)
    valid_dataloader = CIFAR10ChunkDataloader(train=False)

    model = VGG11Async(num_classes=10)
    model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.compile(optimizer, criterion)

    model.fit(train_dataloader=train_dataloader, epochs=10, valid_dataloader=valid_dataloader, batch_size=64)
