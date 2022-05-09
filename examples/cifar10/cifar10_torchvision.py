from time import time

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import vgg11_bn
from torchvision import transforms


def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    print(model)

    val_acc_history = []
    training_start = time()

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            epoch_start = time()

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            num_items = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / num_items
            epoch_acc = running_corrects / num_items
            duration = time() - epoch_start

            print(f'Epoch {epoch+1}, {phase} loss: {epoch_loss:.4f}, '
                  f'{phase} accuracy: {100 * epoch_acc:.2f}% ({duration:.2f}s '
                  f'for {num_items:,} samples {num_items / duration:,.0} fps)', flush=True)

            if phase == 'val':
                val_acc_history.append(epoch_acc)

    time_elapsed = time() - training_start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.2f}%'.format(100*max(val_acc_history)))

    return val_acc_history


if __name__ == '__main__':
    num_workers = 4
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
        'train': DataLoader(train_dataset, batch_size=64, num_workers=num_workers, shuffle=True, drop_last=False,
                            pin_memory=True),
        'val': DataLoader(valid_dataset, batch_size=64, num_workers=num_workers, shuffle=False, drop_last=False,
                          pin_memory=True)
    }

    model = vgg11_bn(num_classes=10)
    model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=10
    )
