import torch
from torchvision.datasets import Country211
from torchvision.transforms import ToTensor
from tqdm import trange


if __name__ == '__main__':
    dataset = Country211(root='./data', split='train', transform=ToTensor())

    sizes = torch.zeros(size=(len(dataset), len(dataset[0][0].shape)), dtype=torch.int32)
    labels = torch.zeros(size=(len(dataset),), dtype=torch.int64)

    for i in trange(len(dataset)):
        features, label = dataset[i]

        sizes[i, :] = torch.Tensor(list(features.shape))
        labels[i] = label

    print(f'Labels check out: {(labels.unique() == torch.arange(labels.max()+1)).all().item()}, '
          f'num_classes={labels.max()+1}')
