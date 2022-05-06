import warnings

from torch.nn import Module
from enum import Enum

from random import sample
from types import SimpleNamespace

from typing import List, Optional, Tuple, Callable
from torch.types import _dtype, _device

import cupy
import torch
from torch import from_numpy, as_tensor, device
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss

from time import time

import multiprocessing as mp

import numpy as np

from .dataloader import ChunkDataloader


class Phase(Enum):
    TRAIN_BEGIN = 1
    TRAIN = 2
    TRAIN_END = 3
    VALID_BEGIN = 4
    VALID = 5
    VALID_END = 6
    TRAIN_EPOCH_BEGIN = 7
    TRAIN_EPOCH_END = 8
    VALID_EPOCH_BEGIN = 9
    VALID_EPOCH_END = 10


# TODO: add example from MNIST /ImageNet... and benchmark vs vanilla torch-vision
# TODO: child processes might not exit nicely from keyboard interrupt
class Model(Module):
    """
    Class designed to manage model training, derives from PyTorch's `Module` class.
    The underlying logic aims at removing data locks by using three asynchronous processes:
    - one for loading the data from hardware storage to CPU memory
    - one for transferring data from GPU to CPU memory
    - one for loading batches of data from GPU memory and training the model
    Important note: this logic was designed for GPU training and requires a CUDA-compatible device to run.
    """
    def __init__(self):
        super().__init__()

        self.train_dataloader = None
        self.valid_dataloader = None
        self.epochs = 0
        self.batch_size = 0

        self.lr_halving_step = 0
        self.bs_doubling_step = 0

        self.loss_function = None
        self.optimizer = None
        self.metrics = None

        self.file_index_queue = None
        self.free_cpu_buffer_index_queue = None
        self.process_cpu_buffer_index_queue = None
        self.free_gpu_buffer_index_queue = None
        self.process_gpu_buffer_index_queue = None
        self.end_epoch_queue = None

        self.num_buffers = dict()
        self.buffer_dtypes = dict()
        self.buffer = dict()

        # TODO: might not be necessary. To be tested and removed if found unnecessary
        self.share_memory()

    def compile(self, optimizer: Optimizer, loss: _Loss, metrics: Optional[List[Callable]] = None):
        """
        Define the optimizer, loss and metrics used for training. The optimizer needs to be initialized with the model's parameters before being passed as a parameter to `compile`.
        :param optimizer: optimizer for the model training
        :param loss: loss function for the model training
        :param metrics: list of metrics to be used for validation
        :return:
        """
        if metrics is not None:
            warnings.WarningMessage('Custom metrics are not yet supported')
        self.optimizer = optimizer
        self.loss_function = loss
        self.metrics = metrics

    def fit(self,
            train_dataloader: ChunkDataloader,
            epochs: int,
            valid_dataloader: Optional[ChunkDataloader] = None,
            batch_size: int = 512,
            lr_halving_step: int = 0,
            bs_doubling_step: int = 0,
            callbacks=None,
            transform=None,
            num_cpu_buffers: int = 4,
            num_gpu_buffers: int = 4,
            cpu_sample_size=None,
            gpu_sample_size=None,
            cpu_dtypes=(torch.uint8, torch.int64),
            gpu_dtypes=(torch.uint8, torch.int64)
            ):
        """
        The `fit` method is used to train the model using the provided dataloaders and training parameters.
        :param train_dataloader: dataloader for training data
        :param epochs: number of epochs to train the model for
        :param valid_dataloader: dataloader for the validation data
        :param batch_size: initial batch size
        :param lr_halving_step: number of epochs for the learning rate to be halved
        :param bs_doubling_step: number of epochs for the batch size to be double
        :param callbacks: list of callbacks
        :param transform: list of transforms
        :param num_cpu_buffers: number of data chunks to simultaneously keep in GPU memory
        :param num_gpu_buffers: number of data chunks to simultaneously keep in GPU memory
        :param cpu_sample_size: size of a sample in CPU memory
        :param gpu_sample_size: size of a sample in GPU memory
        :param cpu_dtypes: data types for (features, labels) on CPU
        :param gpu_dtypes: data types for (features, labels) on GPU
        """
        if callbacks is not None or transform is not None:
            warnings.WarningMessage('Callbacks and transforms are not yet supported')
        if self.optimizer is None or self.loss_function is None:
            raise RuntimeError('You must set the optimizer and loss function via the `compile` method` '
                               'before calling the `fit` method.')
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_halving_step = lr_halving_step
        self.bs_doubling_step = bs_doubling_step

        max_samples_per_buffer = np.max(train_dataloader.num_items())
        sample_size = train_dataloader.sample_size()

        self.num_buffers['cpu'] = num_cpu_buffers
        self.num_buffers['cuda'] = num_gpu_buffers

        self.buffer_dtypes['cpu'] = cpu_dtypes
        self.buffer_dtypes['cuda'] = gpu_dtypes

        for data_type in ['features', 'num_samples', 'labels']:
            self.buffer[data_type] = dict()

        for buffer_device in ['cpu', 'cuda']:
            features_dtype, labels_dtype = self.buffer_dtypes[buffer_device]

            self.buffer['features'][buffer_device] = self.create_buffer(buffer_size=self.num_buffers[buffer_device],
                                                                        chunk_size=(max_samples_per_buffer, sample_size),
                                                                        dtype=features_dtype,
                                                                        device=device(buffer_device))
            self.buffer['num_samples'][buffer_device] = self.create_buffer(buffer_size=self.num_buffers[buffer_device],
                                                                           chunk_size=(1,),
                                                                           dtype=labels_dtype,
                                                                           device=device('cpu'))
            self.buffer['labels'][buffer_device] = self.create_buffer(buffer_size=self.num_buffers[buffer_device],
                                                                      chunk_size=(max_samples_per_buffer,),
                                                                      dtype=torch.int64,
                                                                      device=device(buffer_device))

        self.file_index_queue = mp.Queue()
        self.free_cpu_buffer_index_queue = mp.Queue()
        self.process_cpu_buffer_index_queue = mp.Queue()
        self.free_gpu_buffer_index_queue = mp.Queue()
        self.process_gpu_buffer_index_queue = mp.Queue()

        self.run()

    @staticmethod
    def create_buffer(buffer_size: int, chunk_size: Tuple[int, ...], dtype: _dtype, device: _device):
        """
        Creates a buffer of the specified size and type on the specified device.
        :param buffer_size: size of the buffer
        :param chunk_size: size of each data chunk in the buffer
        :param dtype: data type
        :param device: device on which to store the buffer
        :return buffer
        """
        buffer = [torch.empty(size=chunk_size, dtype=dtype).to(device) for _ in range(buffer_size)]
        # TODO: might not be necessary. To be tested and removed if found unnecessary
        [tensor.share_memory_() for tensor in buffer]

        return buffer

    def run(self):
        """
        Runs the model training with asynchronous processes:
        - data_loader: loads data chunks to a buffer in GPU memory
        - device_transfer: transfers datachunks from CPU memory to GPU memory
        - gpu_process: loads batches from the GPU buffer and trains the model
        The processes inside loops are controlled via several queues:
        - file_index: indexes of the data chunks to load to the buffer
        - free_cpu_buffer_index: cpu buffer indexes where files have already been used and can hold new data
        - process_cpu_buffer_index: cpu buffer indexes to be processed and moved to GPU
        - free_gpu_buffer_index: gpu buffer indexes where files have already been used and can hold new data
        - process_gpu_buffer_index: gpu buffer indexes to be processed and used for training
        :return:
        """
        for i in range(self.num_buffers['cpu']):
            self.free_cpu_buffer_index_queue.put(i)

        for i in range(self.num_buffers['cuda']):
            self.free_gpu_buffer_index_queue.put(i)

        data_loader_process = mp.Process(target=self.data_loader)
        data_loader_process.start()

        data_mover_process = mp.Process(target=self.device_transfer)
        data_mover_process.start()

        training_loop_process = mp.Process(target=self.training_loop)
        training_loop_process.start()

        start = time()

        self.file_index_queue.put(SimpleNamespace(action=Phase.TRAIN_BEGIN))

        for epoch in range(self.epochs):
            self.file_index_queue.put(SimpleNamespace(action=Phase.TRAIN_EPOCH_BEGIN))

            for i in sample(range(self.train_dataloader.num_chunks()), self.train_dataloader.num_chunks()):
                self.file_index_queue.put(SimpleNamespace(action=Phase.TRAIN, index=i))

            self.file_index_queue.put(SimpleNamespace(action=Phase.TRAIN_EPOCH_END))
            self.file_index_queue.put(SimpleNamespace(action=Phase.VALID_EPOCH_BEGIN))

            if self.valid_dataloader is not None:
                for i in range(self.valid_dataloader.num_chunks()):
                    self.file_index_queue.put(SimpleNamespace(action=Phase.VALID, index=i))

            self.file_index_queue.put(SimpleNamespace(action=Phase.VALID_EPOCH_END))

        self.file_index_queue.put(SimpleNamespace(action=Phase.TRAIN_END))

        training_loop_process.join()
        data_mover_process.join()
        data_loader_process.join()

        duration = time() - start
        # TODO: use logging
        print(f'Finished training {len(self.train_dataloader):,} samples for {self.epochs} epochs in '
              f'{duration:,.0f}s (average: {self.epochs * len(self.train_dataloader) / duration:,.0f} fps)')

    def data_loader(self):
        """
        Process that loads data chunks to CPU memory. It reads file indexes from the file_index queue
        and buffer indexes from the free_cpu_buffer_index queue and fills the process_cpu_buffer_index queue
        with the buffer indexes where the files have been loaded.
        :return:
        """
        print(self)

        print(f'Number of training samples: {len(self.train_dataloader):,}')
        print(f'Number of validation samples: {len(self.valid_dataloader):,}')

        while True:
            file_index_queue_object = self.file_index_queue.get()
            action = file_index_queue_object.action

            if not hasattr(file_index_queue_object, 'index'):
                self.process_cpu_buffer_index_queue.put(file_index_queue_object)

                if action == Phase.TRAIN_END:
                    break
            else:
                chunk_index = file_index_queue_object.index
                cpu_buffer_index = self.free_cpu_buffer_index_queue.get()

                if action == Phase.TRAIN:
                    self.load_to_cpu_buffer(self.train_dataloader, chunk_index, cpu_buffer_index)
                elif action == Phase.VALID:
                    self.load_to_cpu_buffer(self.valid_dataloader, chunk_index, cpu_buffer_index)
                else:
                    raise ValueError(f'Unknown action for data_loader process: {action}')

                self.process_cpu_buffer_index_queue.put(SimpleNamespace(action=action, index=cpu_buffer_index))

    def load_to_cpu_buffer(self, dataloader: ChunkDataloader, chunk_index: int, buffer_index: int):
        """
        Loads a chunk of data to the given index of the GPU buffer.
        :param dataloader: chunk dataloader
        :param chunk_index: index of the file to load
        :param buffer_index: index of the buffer to load the file to
        :return:
        """
        num_items, features, labels = dataloader.load_chunk(chunk_index)

        self.buffer['num_samples']['cpu'][buffer_index][0] = num_items
        self.buffer['features']['cpu'][buffer_index][:num_items, :] = from_numpy(features)
        self.buffer['labels']['cpu'][buffer_index][:num_items] = from_numpy(labels)

    def device_transfer(self):
        """
        Process that transfers chunks from CPU buffers to GPU buffers.
        :return:
        """
        while True:
            cpu_buffer_index_queue_object = self.process_cpu_buffer_index_queue.get()
            action = cpu_buffer_index_queue_object.action

            if not hasattr(cpu_buffer_index_queue_object, 'index'):
                self.process_gpu_buffer_index_queue.put(cpu_buffer_index_queue_object)

                if action == Phase.TRAIN_END:
                    break
            else:
                cpu_buffer_index = cpu_buffer_index_queue_object.index
                gpu_buffer_index = self.free_gpu_buffer_index_queue.get()

                if action == Phase.TRAIN or action == Phase.VALID:
                    self.move_buffer(cpu_buffer_index, gpu_buffer_index)
                else:
                    raise ValueError(f'Unknown action for data_loader process: {action}')

                self.free_cpu_buffer_index_queue.put(cpu_buffer_index)
                self.process_gpu_buffer_index_queue.put(SimpleNamespace(action=action, index=gpu_buffer_index))

    def move_buffer(self, cpu_buffer_index, gpu_buffer_index):
        """
        Moves a CPU buffer to GPU.
        :param cpu_buffer_index: index of the CPU buffer to be moved
        :param gpu_buffer_index: index of the GPU buffer to move to
        :return:
        """
        num_items = self.buffer['num_samples']['cpu'][cpu_buffer_index].item()

        self.buffer['num_samples']['cuda'][gpu_buffer_index][0] = num_items
        self.buffer['features']['cuda'][gpu_buffer_index][:num_items, :] = \
            self.buffer['features']['cpu'][cpu_buffer_index][:num_items, :].cuda()
        self.buffer['labels']['cuda'][gpu_buffer_index][:num_items] = \
            self.buffer['labels']['cpu'][cpu_buffer_index][:num_items].cuda()

    def training_loop(self):
        """
        Process that loads the buffered data and trains the model. It gets buffer indexes from
        the process_gpu_buffer_index queue and fills the buffer_index_loading queue with the same index
        when the file has been used.
        :return:
        """
        batch_size = self.batch_size
        epoch = 0
        start = 0
        num_items = 0
        total_loss = 0
        num_correct = 0

        while True:
            gpu_buffer_index_queue_object = self.process_gpu_buffer_index_queue.get()
            action = gpu_buffer_index_queue_object.action

            if action == Phase.TRAIN_EPOCH_BEGIN or action == Phase.VALID_EPOCH_BEGIN:
                start = time()
                num_items = 0
                total_loss = 0
                num_correct = 0

                if action == Phase.TRAIN_EPOCH_BEGIN:
                    self.train()
                    epoch += 1
                elif action == Phase.VALID_EPOCH_BEGIN:
                    self.eval()
            elif action == Phase.TRAIN_EPOCH_END:
                duration = time()-start
                print(f'Epoch {epoch}, training loss: {total_loss / num_items:.4f} ({duration:.2f}s '
                      f'for {num_items:,} samples, {num_items/duration:,.0} fps)', flush=True)

                # TODO: add ceiling for batch size and decrease lr instead
                if self.lr_halving_step:
                    lr = self.lr * 0.5 ** (epoch / self.lr_halving_step)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr

                if self.bs_doubling_step:
                    batch_size = int(self.batch_size * 2. ** (epoch / self.bs_doubling_step))
            elif action == Phase.VALID_EPOCH_END:
                duration = time() - start
                print(f'Epoch {epoch}, validation loss: {total_loss / num_items:.4f}, '
                      f'validation accuracy: {100 * num_correct / num_items:.2f} ({duration:.2f}s '
                      f'for {num_items:,} samples {num_items/duration:,.0} fps)', flush=True)
            elif action == Phase.TRAIN or action == Phase.VALID:
                gpu_buffer_index = gpu_buffer_index_queue_object.index

                num_items_chunk, total_loss_chunk, num_correct_chunk = self.gpu_processing(
                    action, gpu_buffer_index, batch_size)
                num_items += num_items_chunk
                total_loss += total_loss_chunk
                num_correct += num_correct_chunk

                self.free_gpu_buffer_index_queue.put(gpu_buffer_index)
            elif action == Phase.TRAIN_END:
                break

    def gpu_processing(self, action, gpu_buffer_index, batch_size):
        """
        Trains on the data specified by the given buffer index using the given batch size.
        :param action: queue message, can be "TRAIN" or "VALID"
        :param gpu_buffer_index: index of the data in the GPU buffer
        :param batch_size: current batch size
        :return:
        """
        num_items = self.buffer['num_samples']['cuda'][gpu_buffer_index].item()
        indices = self.shuffled_indices(num_items, batch_size)

        total_loss = 0
        num_correct = 0

        for index in indices:
            index = as_tensor(index, device=device('cuda'))

            features = self.buffer['features']['cuda'][gpu_buffer_index][index].float()
            labels = self.buffer['labels']['cuda'][gpu_buffer_index][index]

            outputs = self(features)
            loss = self.loss_function(outputs, labels)

            _, preds = torch.max(outputs, 1)

            # statistics
            total_loss += loss.item() * features.size(0)
            num_correct += torch.sum(preds == labels)

            if action == Phase.TRAIN:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return num_items, total_loss, num_correct

    @staticmethod
    def shuffled_indices(num_items, batch_size):
        """
        Method that samples indexes and reshapes the resulting array to num_batches x batch_size.
        :param num_items: number of items to sample from
        :param batch_size: current batch size
        :return: sampled indices with second dimension equal to batch_size
        """
        num_items_reduced = (num_items // batch_size) * batch_size

        indices = cupy.random.choice(num_items, num_items, replace=False)
        batch_indices = list(cupy.reshape(indices[:num_items_reduced], (-1, batch_size)))

        if num_items > num_items_reduced:
            batch_indices += [indices[num_items_reduced:]]

        return batch_indices
