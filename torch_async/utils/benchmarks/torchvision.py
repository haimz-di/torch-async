from time import time

import numpy as np
import torch


def run_benchmark(model_constructor, model_kwargs, dataloaders, criterion, optimizer_constructor, optimizer_kwargs,
                  num_epochs=10, num_runs=10):
    best_val_acc = []
    training_time = []

    for _ in range(num_runs):
        model = model_constructor(**model_kwargs)
        model.cuda()

        optimizer = optimizer_constructor(model.parameters(), **optimizer_kwargs)

        results = train_model(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs
        )

        best_val_acc.append(results[0])
        training_time.append(results[1])

    print(f'Validation accuracy: {np.mean(best_val_acc):.2f} +- {np.std(best_val_acc):.2f}')
    print(f'Training time: {np.mean(training_time):.2f} +- {np.std(training_time):.2f}')


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
                running_corrects += torch.sum(preds == labels.data).item()

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
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {100*np.max(val_acc_history):.2f}%')

    return 100 * np.max(val_acc_history), time_elapsed
