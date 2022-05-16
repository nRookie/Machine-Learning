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