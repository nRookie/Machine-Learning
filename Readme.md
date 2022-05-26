## Errors

``` shell
Traceback (most recent call last):
  File "main.py", line 203, in <module>
    output = G.forward(generate_random_seed(100))
  File "main.py", line 128, in forward
    return self.model(inputs)
  File "/root/miniconda3/envs/pytorch/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "/root/miniconda3/envs/pytorch/lib/python3.7/site-packages/torch/nn/modules/container.py", line 141, in forward
    input = module(input)
  File "/root/miniconda3/envs/pytorch/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "/root/miniconda3/envs/pytorch/lib/python3.7/site-packages/torch/nn/modules/linear.py", line 103, in forward
    return F.linear(input, self.weight, self.bias)
  File "/root/miniconda3/envs/pytorch/lib/python3.7/site-packages/torch/nn/functional.py", line 1848, in linear
    return torch._C._nn.linear(input, weight, bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x100 and 1x200)
(pytorch) [root@10-9-107-255 Handwritten]# python3 main.py 
torch.Size([100])
Traceback (most recent call last):
  File "main.py", line 202, in <module>
    output = G.forward(generate_random_seed(100))
  File "main.py", line 128, in forward
    return self.model(inputs)
  File "/root/miniconda3/envs/pytorch/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "/root/miniconda3/envs/pytorch/lib/python3.7/site-packages/torch/nn/modules/container.py", line 141, in forward
    input = module(input)
  File "/root/miniconda3/envs/pytorch/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "/root/miniconda3/envs/pytorch/lib/python3.7/site-packages/torch/nn/modules/linear.py", line 103, in forward
    return F.linear(input, self.weight, self.bias)
  File "/root/miniconda3/envs/pytorch/lib/python3.7/site-packages/torch/nn/functional.py", line 1848, in linear
    return torch._C._nn.linear(input, weight, bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x100 and 1x200)
```

``` python
# generator
        self.model = nn.Sequential(
            nn.Linear(100, 200), # note the shape
            nn.LeakyReLU(0.02),

            nn.LayerNorm(200),

            nn.Linear(200, 784),
            nn.Sigmoid()
        )
##3 should match
class Discriminator(nn.Module):
    def __init__(self):
        # initialise parent pytorch class
        super().__init__()

        # define neural network layers
        self.model = nn.Sequential(
            nn.Linear(784, 200), # note the shape
            nn.LeakyReLU(0.02),

            nn.LayerNorm(200),

            nn.Linear(200, 1),
            nn.Sigmoid()
        )

```



# errors

https://kontext.tech/article/915/torchvision-error-could-not-find-module-imagepyd



# install 
https://www.yisu.com/zixun/608015.html

https://discuss.pytorch.org/t/pytorch-cuda-11-6/149647




# celebA
https://www.pythonfixing.com/2022/02/fixed-dataset-not-found-or-corrupted.html




## cuda not enabled

``` shell
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```





## RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking a

### move code to cuda
``` python
if torch.cuda.is_available():
  torch.set_default_tensor_type(torch.cuda.FloatTensor)
  print("using cuda:", torch.cuda.get_device_name(0))
  pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D.to(device)
```


### We can see that there is a change in nvidia-smi

``` shell
(pytorch-pip) [root@10-9-151-144 HumanFace]# nvidia-smi
Thu May 26 21:10:41 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:03.0 Off |                    0 |
| N/A   34C    P0    25W /  70W |      0MiB / 15360MiB |     10%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
(pytorch-pip) [root@10-9-151-144 HumanFace]# ls
celeba  celeba.backup  celeba_dataset  celebadataset.py  Discriminator.py  img_align_celeba  looking_data.py  main.py  __pycache__
(pytorch-pip) [root@10-9-151-144 HumanFace]# nvidia-smi 
Thu May 26 21:28:07 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:03.0 Off |                    0 |
| N/A   54C    P0    71W /  70W |   1367MiB / 15360MiB |     95%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     28466      C   python3                          1365MiB |
+-----------------------------------------------------------------------------+
(pytorch-pip) [root@10-9-151-144 HumanFace]# 

```

