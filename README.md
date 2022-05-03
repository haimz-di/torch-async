# torch-async
Torch Async performs asynchronous data loading and model training for PyTorch.

It was design to overcome the limitations of the sequential nature of PyTorch standard training loop by removing locks in the data loading and model training process.

The standard data loading and training step is displayed in Figure 1 (left): 
1) loading samples from hardware storage (SSD, network storage...) to CPU memory,
2) pre-processing data batch (unpacking data, normalization...) and moving it to GPU memory,
3) perform training step on the model.
As each step is performed in a synchronous way (even if steps 1 and 2 use the multi-processed dataloader from PyTorch the batch collate and training steps are done synchronously), locks occur in the process and resources are left unused.

What Torch Async does is to decouple these different steps by running them from different processes as shown in Figure 1 (right):
1) loading samples from hardware storage (SSD, network storage...) to the CPU buffer,
2) pre-processing data batch (unpacking data, normalization...) and moving it to the GPU buffer,
3) perform training step on the model with data from the GPU buffer.

<p align="center">
    <img src="images/sequential_process.svg" />
    <img src="images/async_process.svg" />
</p>
<p align = "center">
Figure 1: Standard sequential training flow (left) and decoupled multi-process training flow (right). 
</p>

A more detailed explanation on how this asynchronous data loading and model training is performed is shown in Figure 2.
Five queues are used to sent messages:
- `loadable chunks`: is filled with the list of data chunks to be loaded and processed,
- `free CPU buffer`: contains the indices of free CPU buffers which can be loaded with new data chunks,
- `process CPU buffer`: once a data chunk is loaded to a CPU buffer, a message is sent via this queue to notify that this data chunk is ready to be processed,
- `free GPU buffer`: contains the indices of free GPU buffer which can be loaded with data from a CPU buffer,
- `process GPU buffer`: once a data chunk is loaded to a GPU buffer, a message is sent via this queue to notify that this data chunk is ready to be processed.
Three asynchronous processes use the above queues to communicate:
1) the `data loader` process reads messages from the `loadable chunks` queue (which lists the chunks to load for each epoch) and the `free CPU buffer` queue (list of available CPU buffers). Once a chunk is loaded into a CPU buffer, a message is sent to the `process CPU buffer` queue,
2) the `data mover` process reads messages from the `process CPU buffer` queue and moves the data to a free GPU buffer using the `free GPU buffer` queue. Once a chunk is moved to a GPU buffer, a message is sent to the `free CPU buffer` queue and the `process GPU buffer` queue, 
3) the `GPU processing` process reads messages from the `process GPU buffer` queue and once the chunk has been processed writes it to the `free GPU buffer` queue.
4) 
<p align="center">
    <img src="images/async_flow.svg" />
</p>
<p align = "center">
Figure 2: Detailed description of the data loading and model training asynchronous process flow. 
</p>
