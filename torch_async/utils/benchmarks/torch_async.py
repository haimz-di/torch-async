from glob import glob
import numpy as np


def run_benchmark(model_constructor, model_kwargs, dataloaders, criterion, optimizer_constructor, optimizer_kwargs,
                  num_epochs=10, batch_size=64, num_runs=10):
    for _ in range(num_runs):
        model = model_constructor(**model_kwargs)
        model.cuda()

        optimizer = optimizer_constructor(model.parameters(), **optimizer_kwargs)

        model.compile(optimizer, criterion)

        model.fit(train_dataloader=dataloaders['train'], epochs=num_epochs, valid_dataloader=dataloaders['val'],
                  batch_size=batch_size)

    best_val_acc = []
    training_time = []

    for file_path in glob('torchasync-*.txt'):
        with open(file_path, 'rt') as fp:
            result = [float(val.strip()) for val in fp.read().split(',')]

            best_val_acc.append(result[0])
            training_time.append(result[1])

    print(f'Validation accuracy: {np.mean(best_val_acc):.2f} +- {np.std(best_val_acc):.2f}')
    print(f'Training time: {np.mean(training_time):.2f} +- {np.std(training_time):.2f}')
