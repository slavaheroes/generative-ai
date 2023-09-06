'''
    Inspired: https://huggingface.co/blog/annotated-diffusion
    and https://nn.labml.ai/normalization/weight_standardization/index.html
    and https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/simple_diffusion.py
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from einops import rearrange

class Conv2d(nn.Conv2d): 
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 weight_standard):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.weight_standard = weight_standard
     
    def forward(self, x):
        if self.weight_standard:
            eps = 1e-5
            
            if x.dtype==torch.float16:
                eps = 1e-3
            
            weight_shape = self.weight.shape
            weight = self.weight.view(weight_shape[0], -1)
            
            var, mean = torch.var_mean(
                weight,
                dim=1,
                keepdim=True
            )          
            weight = (weight - mean)/torch.sqrt(var+eps)
            
            weight = weight.view(weight_shape)
            
        else:
            weight = self.weight
        
        return F.conv2d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )
            
class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        groups
    ):
        super().__init__()
        
        self.time_embed_mlp = nn.Sequential(
            nn.SiLU(),
            nn.LazyLinear(out_channels*2)
        )
        
        self.act_func = nn.SiLU()
        
        self.block1 = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, weight_standard=False),
            nn.GroupNorm(groups, out_channels)
        )
        
        # if compared with WideResNet
        # deep factor is 2
        
        self.block2 = nn.Sequential(
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, weight_standard=False),
            nn.GroupNorm(groups, out_channels)
        )
        
        if in_channels!=out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                   weight_standard=False)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x, t):
        time_embed = self.time_embed_mlp(t)
        time_embed = rearrange(time_embed, "b c -> b c 1 1")
        scale, shift = time_embed.chunk(2, dim=1)
        
        out = self.block1(x)
        out = out*(scale + 1) + shift
        out = self.act_func(out)
        
        out = self.act_func(self.block2(out))
        
        return out + self.shortcut(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, n_heads, dim_head):
        super().__init__()
        hidden_dim = dim_head*n_heads
        self.n_heads = n_heads
        self.dim_head = dim_head
        
        self.qkv_mapping = Conv2d(in_channels, 3*hidden_dim, kernel_size=1, padding=0, stride=1, weight_standard=False)
        self.out_mapping = Conv2d(hidden_dim, in_channels, kernel_size=1, padding=0, stride=1, weight_standard=False)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        _, _, h, w = x.shape
        qkv = self.qkv_mapping(x).chunk(3, dim=1)
        
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.n_heads), qkv)
        attention = self.softmax(torch.einsum('bhdi, bhdj -> bhij', (q, k)) /(self.dim_head ** 0.5))
        out = torch.einsum('bhij, bhdj -> bhid', (attention, v))
        
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.out_mapping(out)
    
class AttentionBlock(nn.Module):
    def __init__(self, dim, n_heads, dim_head):
        super().__init__()
        self.msa = MultiHeadAttention(dim, n_heads, dim_head)
        self.norm = nn.GroupNorm(1, dim)
    
    def forward(self, x):
        x1 = self.norm(x)
        return self.msa(x1) + x

class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel, with_stride=False):
        super().__init__()
        if with_stride:
            self.down = Conv2d(in_channel, out_channel, kernel_size=1, stride=2, padding=0, weight_standard=False)
        else:
            self.down = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
                Conv2d(4*in_channel, out_channel, kernel_size=1, padding=0, stride=1, weight_standard=False)
            )        

    def forward(self, x):
        return self.down(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, is_transpose=True):
        super().__init__()
        if is_transpose:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, weight_standard=False)
            )
    
    def forward(self, x):
        return self.up(x)
    
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim, learnable=False):
        super().__init__()
        self.learnable = learnable
        if learnable:
            self.weights = nn.Parameter(torch.randn(dim//2))
        
        self.dim = dim
    
    def forward(self, time):
        half_dim = self.dim//2
        embeddings = math.log(10000) / (half_dim - 1)
        
        if self.learnable:
            freqs = rearrange(self.weights, 'd -> 1 d')
        else:
            freqs = torch.arange(half_dim, device=time.device)
        
        embeddings = torch.exp(freqs * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
        

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        dim=64,
        width_factor=[1, 2, 4, 8], # width = 1
        groups=8,
        downsample_w_stride=True,
        upsample_w_transpose=True,
        learnable_pos_embed=False,
        n_heads=4,
        dim_head=32
    ):
        super().__init__()
        
        time_dim = 4*dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim, learnable=learnable_pos_embed),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.first_conv = Conv2d(in_channels, out_channels=dim*width_factor[0], kernel_size=1, padding=0, stride=1, weight_standard=False)
        
        # downsampling
        self.downs = nn.ModuleList([])
        for i in range(len(width_factor)-1):
            in_ch, out_ch = dim*width_factor[i], dim*width_factor[i+1]
            
            self.downs.append(nn.ModuleList([
                ResNetBlock(in_channels=in_ch, out_channels=in_ch, groups=groups),
                ResNetBlock(in_channels=in_ch, out_channels=out_ch, groups=groups),
                AttentionBlock(out_ch, n_heads, dim_head),
                nn.Identity() if (i==len(width_factor)-2) else Downsample(in_channel=out_ch, out_channel=out_ch, with_stride=downsample_w_stride)
            ]))
        
        # middle bottleneck
        self.middle = nn.ModuleList([
            ResNetBlock(dim*width_factor[-1], dim*width_factor[-1], groups=groups),
            AttentionBlock(dim*width_factor[-1], n_heads, dim_head),
            ResNetBlock(dim*width_factor[-1], dim*width_factor[-1], groups=groups)
        ])
        
        # upsampling
        self.ups = nn.ModuleList([])
        for i in range(len(width_factor)-1, 0, -1):
            in_ch, out_ch = dim*width_factor[i], dim*width_factor[i-1]
            
            self.ups.append(nn.ModuleList([
                ResNetBlock(in_channels=2*in_ch, out_channels=out_ch, groups=groups),
                ResNetBlock(in_channels=2*out_ch, out_channels=out_ch, groups=groups),
                AttentionBlock(out_ch, n_heads, dim_head),
                nn.Identity() if i==1 else Upsample(in_channels=out_ch, out_channels=out_ch, is_transpose=upsample_w_transpose)
            ]))

        self.final_res = ResNetBlock(in_channels=2*out_ch, out_channels=out_ch, groups=groups)
        self.final_conv = Conv2d(in_channels=out_ch, out_channels=in_channels, kernel_size=1, stride=1, padding=0, weight_standard=False)
        
        
    def forward(self, x, t):
        x = self.first_conv(x)
        
        x_init = x.clone()
        
        t = self.time_mlp(t)
        h = []

        # down
        
        for block1, block2, attn, down in self.downs:
            x = block1(x, t)
            h.append(x)
            
            x = block2(x, t)
            
            x = attn(x)
            h.append(x)
            
            x = down(x)
            
            
        
        # mid
        block1, attn, block2 = self.middle
        x = block1(x, t)
        x = attn(x)
        x = block2(x, t)
        
        # up
        for block1, block2, attn, up in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
                        
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = up(x)
        
        x = self.final_res(torch.cat((x, x_init), dim=1), t)
        
        x = self.final_conv(x)
        return x
        
           
if __name__=="__main__":
    print("Test")
    x = torch.randn((1, 1, 32, 32))
    t = torch.Tensor([1])
    m = UNet(dim=32, downsample_w_stride=False, upsample_w_transpose=False, width_factor=(1, 2, 4), in_channels=1)
    print(x.min(), x.max())
    print(m(x, t).unique())
