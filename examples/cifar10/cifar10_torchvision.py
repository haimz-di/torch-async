import sys
sys.path.insert(0, '../..')

from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import vgg11_bn
from torchvision import transforms

from torch_async.utils.benchmarks.torchvision import run_benchmark


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--num-runs', type=int, default=10, help='Number of experiments to run')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs to run each experiment')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for the Dataloader')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)

    transform_train = transforms.Compose(
        [
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std),
        ]
    )
    transform_valid = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(mean, std),
        ]
    )
    train_dataset = CIFAR10(root='./data', train=True, transform=transform_train, download=True)
    valid_dataset = CIFAR10(root='./data', train=False, transform=transform_valid, download=True)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                            drop_last=False, pin_memory=True),
        'val': DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                          drop_last=False, pin_memory=True)
    }

    model_constructor = vgg11_bn
    model_kwargs = {'num_classes': 10}

    criterion = torch.nn.CrossEntropyLoss()
    optimizer_constructor = torch.optim.SGD
    optimizer_kwargs = {'lr': 0.001, 'momentum': 0.9}

    run_benchmark(
        model_constructor=model_constructor,
        model_kwargs=model_kwargs,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer_constructor=optimizer_constructor,
        optimizer_kwargs=optimizer_kwargs,
        num_epochs=args.num_epochs,
        num_runs=args.num_runs
    )
