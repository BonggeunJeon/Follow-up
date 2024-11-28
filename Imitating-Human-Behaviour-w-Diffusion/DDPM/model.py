import torch
import math
from torch import nn
from torch.nn import init
from torch.nn import functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class TimeEmbedding(nn.Module): # Tiem Embedding 이 뭐임? 즉 어떤 특성을 가지고 있음?
    """ 
        I think, time embedding can be used at each denoising step. Because we have to know how much noisy has been added.
        I mean, If we don't have time embedding, we don't know the current noisy levle.
        (내 생각에, Time embedding 각 denoising step에서 사용될 수 있어, 왜냐하면 각 denoising step 마다 얼마나 noisy가 생겼는지를 알아야 하니까
        그니까 만약에 우리가 타임 임베딩을 안하면, 현재의 noisy 수준을 몰라)
    """
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb) # exponential 의 어떤 성질 때문에 여기다 이걸 썼을까?
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2] 
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)
        
        """ 
            There are two variables which the tensor value can be put into. And these variables are initialized first.
            (Tensor 값들이 들어가는 변수가 emb 와 pos 가 있는데, 그 구조를 초기화를 먼저 시킴)
        """
        
        self.timeembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()
        
        def initialize(self):
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    init.xavier_uniform_(module.weight)
                    init.zeros_(module.bias)
        
        def forward(self, t):
            emb = self.timeembedding(t)
            return emb