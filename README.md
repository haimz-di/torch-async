# torch-async
Torch Async performs asynchronous data loading and model training for PyTorch.

It was design to overcome the limitations of the sequential nature of PyTorch standard training loop by removing locks in the data loading and model training process.

The standard data loading and training step is displayed in Figure 1 (left): 
1) loading samples from hardware storage (hard drive, network storage...) to system memory
2) pre-processing samples (unpacking data, normalization...) 
3) transfer pre-processed samples to GPU memory and 
4) perform training step on the model,
As each step is performed in a synchronous way (even if steps 1 and 2 use the multi-processed dataloader from PyTorch the batch collate and training steps are done synchronously), locks occur in the process and resources are left unused.

What Torch Async does is to decouple these different steps by running them from different processes as shown in Figure 1 (right).

<p align="center">
    <img src="images/sequential_process.svg" />
    <img src="images/async_process.svg" />
</p>
<p align = "center">
Figure 1: Standard sequential training flow (left) and decoupled multi-process training flow (right). 
</p>


