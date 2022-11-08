horovodrun -np 4 python mnist_horovod.py > mnist_horovod.txt
torchrun --nproc_per_node=2 --nnodes=1 mnist_torch.py
