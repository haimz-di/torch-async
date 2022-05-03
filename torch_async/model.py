from torch.nn import Module
from enum import Enum

from random import sample
from types import SimpleNamespace

from typing import List, Optional, Dict, Tuple

import cupy
import torch
from torch import from_numpy, as_tensor, device

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
# TODO: make compatible for generic metrics
# TODO: test torchmetrics
# TODO: implement predict and evaluate functions
# TODO: add support for memory pinning (both CPU and GPU)
# TODO: add support for resuming from existing checkpoint
class Model(Module):
    """
    Class designed to manage model training.
    The underlying logic aims at removing data locks by using three asynchronous processes:
    - one for loading the data from disk to CPU memory
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
        self.cpu_buffer_index_process_queue = None
        self.free_gpu_buffer_index_queue = None
        self.gpu_buffer_index_process_queue = None
        self.end_epoch_queue = None

        self.num_buffers = dict()
        self.buffer = dict()

        # TODO: might not be necessary. To be tested and removed if found unnecessary
        self.share_memory()

    def compile(self, optimizer, loss, metrics=None):
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
            gpu_sample_size=None):
        """
        :param train_dataloader:
        :param epochs:
        :param valid_dataloader:
        :param batch_size:
        :param lr_halving_step:
        :param bs_doubling_step:
        :param callbacks:
        :param transform:
        :param num_cpu_buffers: number of data chunks to simultaneously keep in GPU memory
        :param num_gpu_buffers: number of data chunks to simultaneously keep in GPU memory
        :param cpu_sample_size:
        :param gpu_sample_size:
        """
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

        for data_type in ['features', 'num_samples', 'labels']:
            self.buffer[data_type] = dict()

        for device in ['cpu', 'cuda']:
            self.buffer['features'][device] = self.create_buffer(buffer_size=self.num_buffers[device],
                                                                 chunk_size=(max_samples_per_buffer, sample_size),
                                                                 dtype=torch.uint8,
                                                                 device=device)
            self.buffer['num_samples'][device] = self.create_buffer(buffer_size=self.num_buffers[device],
                                                                    chunk_size=(1,),
                                                                    dtype=torch.int32,
                                                                    device=device)
            self.buffer['labels'][device] = self.create_buffer(buffer_size=self.num_buffers[device],
                                                               chunk_size=(max_samples_per_buffer, 1),
                                                               dtype=torch.uint8,
                                                               device=device)

        self.file_index_queue = mp.Queue()
        self.free_cpu_buffer_index_queue = mp.Queue()
        self.cpu_buffer_index_process_queue = mp.Queue()
        self.free_gpu_buffer_index_queue = mp.Queue()
        self.gpu_buffer_index_process_queue = mp.Queue()

        self.run()

    @staticmethod
    def create_buffer(buffer_size: int, chunk_size: Tuple[int, ...], dtype, device):
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
        - data_mover: moves datachunks from CPU memory to GPU memory
        - gpu_process: loads batches from the GPU buffer and trains the model
        Additionally a process called "process_end_epoch" is dedicated to end-of-epoch callback calls and metric
        computations.
        The processes inside loops are controlled via several queues:
        - file_index: contains the indexes of the NumPy mmap files to load to the buffer
        - free_cpu_buffer_index: contains the cpu buffer indexes where files have already been used and can hold new data
        - free_gpu_buffer_index: contains the gpu buffer indexes where files have already been used and can hold new data
        - gpu_buffer_index_process: contains the gpu buffer indexes where files have been loaded for training
        - end_epoch: a message is sent to this queue at the end of each epoch
        :return:
        """
        for i in range(self.num_buffers['cpu']):
            self.free_cpu_buffer_index_queue.put(i)

        for i in range(self.num_buffers['cuda']):
            self.free_gpu_buffer_index_queue.put(i)

        data_loader_process = mp.Process(target=self.data_loader)
        data_loader_process.start()

        data_mover_process = mp.Process(target=self.data_mover)
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
        Method that loads NumPy mmap files from disk to GPU memory. It reads file indexes from the file_index queue
        and buffer indexes from the buffer_index_loading queue and fills the buffer_index_process queue with the buffer
        indexes where the files have been loaded.
        :return:
        """
        print(self)

        print(f'Number of training samples: {len(self.train_dataloader):,}')
        print(f'Number of validation samples: {len(self.valid_dataloader):,}')

        while True:
            file_index_queue_object = self.file_index_queue.get()
            action = file_index_queue_object.action

            if not hasattr(file_index_queue_object, 'index'):
                self.cpu_buffer_index_process_queue.put(file_index_queue_object)

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

                self.cpu_buffer_index_process_queue.put(SimpleNamespace(action=action, index=cpu_buffer_index))

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
        self.buffer['labels']['cpu'][buffer_index][:num_items, :] = from_numpy(labels)

    def data_mover(self):
        while True:
            cpu_buffer_index_queue_object = self.cpu_buffer_index_process_queue.get()
            action = cpu_buffer_index_queue_object.action

            if not hasattr(cpu_buffer_index_queue_object, 'index'):
                self.gpu_buffer_index_process_queue.put(cpu_buffer_index_queue_object)

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
                self.gpu_buffer_index_process_queue.put(SimpleNamespace(action=action, index=gpu_buffer_index))

    def move_buffer(self, cpu_buffer_index, gpu_buffer_index):
        num_items = self.buffer['num_samples']['cpu'][cpu_buffer_index][0]

        self.buffer['num_samples']['cuda'][gpu_buffer_index][0] = num_items
        self.buffer['features']['cuda'][gpu_buffer_index][:num_items, :] = \
            self.buffer['features']['cpu'][cpu_buffer_index][:num_items, :].cuda()
        self.buffer['labels']['cuda'][gpu_buffer_index][:num_items, :] = \
            self.buffer['labels']['cpu'][cpu_buffer_index][:num_items, :].cuda()

    def training_loop(self):
        """
        Process that loads the buffered data and trains the model. It gets buffer indexes from the buffer_index_process
        queue and fills the buffer_index_loading queue with the same index when the file has been used.
        :return:
        """
        batch_size = self.batch_size
        epoch = 0
        start = 0
        num_items = 0
        total_loss = 0
        num_correct = 0

        while True:
            gpu_buffer_index_queue_object = self.gpu_buffer_index_process_queue.get()
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
        Train on the data specified by the given buffer index using the given batch size.
        :param action:
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

            features = self.buffer['features']['cuda'][gpu_buffer_index][index]
            features = as_tensor(cupy.unpackbits(cupy.asarray(features))
                                       .reshape(features.shape[0], -1), device=device('cuda')).float()
            labels = self.buffer['labels']['cuda'][gpu_buffer_index][index].float()

            outputs = self(features)
            loss = self.loss_function(outputs, labels)

            total_loss += loss.item() * len(labels)

            if action == Phase.TRAIN:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            elif action == Phase.VALID:
                num_correct += ((outputs > 0.5).float() == labels).sum().float().item()

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
