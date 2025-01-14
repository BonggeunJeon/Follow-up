import torch
import torch.nn.functional as F

import numpy as np

from torch import nn
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary


'''
    The number of pictures : 8
    The number of channels per picture : 3
    Width & Height per channel : 224
'''
x = torch.randn(8, 3, 224, 224)

#print('x : ', x.shape)

'''
    einops <- Readable & intuitive operations for the process of transforming tensors
    
    b : batch
    h : height
    w : width
    c : channel
'''
input_tensor = [np.random.randn(30, 40, 3) for _ in range(32)]


output_tensor = rearrange(input_tensor, "b h w c -> b h w c")
output_tensor_1 = rearrange(input_tensor, "b h w c -> (b h) w c")
output_tensor_2 = rearrange(input_tensor, "b h w c -> (b w) h c")
output_tensor_3 = rearrange(input_tensor, "b h w c -> b c h w")
output_tensor_4 = rearrange(input_tensor, "b h w c -> b (c h w)")

print('input_tensor : ', torch.Tensor(input_tensor).shape)
print('output_tensor_1 : ', torch.Tensor(output_tensor_1).shape)
print('output_tensor_2 : ', torch.Tensor(output_tensor_2).shape)
print('output_tensor_3 : ', torch.Tensor(output_tensor_3).shape)
print('output_tensor_3 : ', torch.Tensor(output_tensor_4).shape)
#patch_size = 16
#patch_size = 16
#patch_size = 16
#patches = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)


