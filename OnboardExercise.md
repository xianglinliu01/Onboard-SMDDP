# Onboard play ground
## Author information
 * @author Xianglin Liu
 * @email xianglinliu01@gmail.com
 * @create date 2022-11-04 15:37:26
 * @modify date 2022-11-04 15:37:26

## Exercises
* Check the quip-amazon file : [Herring New Team Member Onboarding](https://quip-amazon.com/7vTJAL7FQPKu/Herring-New-Team-Member-Onboarding)
### Problem 8: Train a simple model using PyTorch
* Create the conda environment and install pytorch:
  ```bash
  conda create -name pytorch
  conda install pytorch ipython  torchvision
  ```
* Check the cuda version has been installed in python:
    ```bash
    In [1]: import torch

    In [2]: torch.cuda.is_available()
    Out[2]: True

    In [3]: torch.cuda.get_device_name(0)
    Out[3]: 'NVIDIA A100-SXM4-40GB'

    In [4]: torch.cuda.device_count()
    Out[4]: 8
    ```
* Copy the code at https://github.com/pytorch/examples/tree/master/mnist and run it: ```python main.py > output.txt```; the ouput is saved.
* Export the conda package dependency as a yml file ```conda env export > env_pytorch_mnist.yml```.

### Problem 9: Towards Distributed Data Parallel Training
