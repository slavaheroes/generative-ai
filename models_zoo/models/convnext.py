'''
This file contains implementation of https://arxiv.org/abs/2201.03545

Referenced from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

Train in fp32 precision
'''
import torch
import torch.nn as nn

from timm.models.layers import DropPath
import einops

        
class PatchifyStem(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=4)
        self.ln = nn.LayerNorm(out_channels)
    
    def forward(self, x):
        out = self.conv(x)
        out = einops.rearrange(out, 'b c h w -> b h w c')
        out = self.ln(out)
        out = einops.rearrange(out, 'b h w c -> b c h w')
        return out


class Downsample_Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.ln = nn.LayerNorm(out_channels)
    
    def forward(self, x):
        out = self.conv(x)
        out = einops.rearrange(out, 'b c h w -> b h w c')
        out = self.ln(out)
        out = einops.rearrange(out, 'b h w c -> b c h w')
        return out


class ConvNeXt_Block(nn.Module):
    def __init__(
        self,
        dim,
        drop_path,
    ):
        super().__init__()
        
        self.conv_d7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim)
        self.ln = nn.LayerNorm(dim)
        self.conv_d1_1 = nn.Conv2d(dim, 4*dim, kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU()
        self.conv_d1_2 = nn.Conv2d(4*dim, dim, kernel_size=1, stride=1, padding=0)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x):
        out = self.conv_d7(x)
        out = einops.rearrange(out, 'b c h w -> b h w c')
        out = self.ln(out)
        out = einops.rearrange(out, 'b h w c -> b c h w')
        
        out = self.conv_d1_2(self.act(self.conv_d1_1(out)))
        
        return out + self.drop_path(x)
    
    
class ConvNeXt(nn.Module):
    def __init__(self,
                 in_channels=3,
                 num_classes=10,
                 channels=[96, 192, 384, 768],
                 num_blocks=[3, 3, 9, 3],
                 drop_path_rate=0.3,
                 ):
        super().__init__()
        
        self.patchify = PatchifyStem(in_channels=in_channels, out_channels=channels[0])
        
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]
        
        self.blocks = nn.ModuleList()
        
        for idx in range(len(num_blocks)):
            
            self.blocks.append(
                nn.Sequential(*[
                    ConvNeXt_Block(channels[idx], drop_path=dp_rates[sum(num_blocks[:idx]) + j])
                    for j in range(num_blocks[idx])
                ])
            )
            # downsample block
            if idx != len(num_blocks)-1:
                self.blocks.append(Downsample_Block(in_channels=channels[idx], out_channels=channels[idx+1]))
        
        self.ln = nn.LayerNorm(channels[-1])
        
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)
        
    def forward(self, x):
        x = self.patchify(x)
        for block in self.blocks:
            x = block(x)
        
        x = x.view(-1, x.shape[1])
        out = self.fc(x)
        return out
            
        

if __name__=="__main__":
    x = torch.Tensor(10, 3, 32, 32)
    m = ConvNeXt()
    print(m(x).shape)
    print(sum([p.numel() for p in m.parameters()]))