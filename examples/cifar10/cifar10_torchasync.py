import sys
sys.path.insert(0, '../..')

import multiprocessing as mp

import torch
from argparse import ArgumentParser

from torch_async.utils.benchmarks.torch_async import run_benchmark

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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--num-runs', type=int, default=10, help='Number of experiments to run')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs to run each experiment')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')

    return parser.parse_args()


if __name__ == '__main__':
    # TODO: test solutions from https://github.com/pytorch/pytorch/issues/3492
    mp.set_start_method('spawn')

    args = parse_args()

    dataloaders = {
        'train': CIFAR10ChunkDataloader(train=True),
        'val': CIFAR10ChunkDataloader(train=False)
    }

    model_constructor = VGG11Async
    model_kwargs = {'num_classes': 10}

    criterion = torch.nn.CrossEntropyLoss()

    optimizer_constructor = torch.optim.SGD
    optimizer_kwargs = {'lr': 0.001, 'momentum': 0.9}

    run_benchmark(model_constructor=model_constructor,
                  model_kwargs=model_kwargs,
                  dataloaders=dataloaders,
                  criterion=criterion,
                  optimizer_constructor=optimizer_constructor,
                  optimizer_kwargs=optimizer_kwargs,
                  num_epochs=args.num_epochs,
                  batch_size=args.batch_size,
                  num_runs=args.num_runs
                  )
