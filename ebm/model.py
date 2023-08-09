'''
This file contains implementation of https://arxiv.org/abs/1605.07146
The configuration as follows:
    - block type B(3,3)
    - no dropout
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Conv2d):
    '''
    Inspired: https://huggingface.co/blog/annotated-diffusion
    and https://nn.labml.ai/normalization/weight_standardization/index.html
    ''' 
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
            

class WideResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        deep_factor=2,
        act_func=nn.ReLU,
        norm_fun=nn.Identity,
        weight_standardization=False
    ):
        super().__init__()
        
        self.first_conv = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, weight_standard=weight_standardization),
            norm_fun(out_channels),
            act_func()
        )
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, weight_standard=weight_standardization),
                norm_fun(out_channels),
                act_func()
            ) for _ in range(deep_factor-1)
        ])
        
        if in_channels!=out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                   weight_standard=False)
        else:
            self.shortcut = nn.Identity()
        
        self.weight_standard = weight_standardization
    
    def forward(self, x):
        out = self.first_conv(x)
        x = self.shortcut(x)
        for idx, layer in enumerate(self.blocks, 1):
            if idx%2!=0:
                out = layer(out)
            else:
                out = layer(out) + x 
                x = out
                
        return out
        
class WideResNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        feature_dims=[16, 32, 64],
        num_classes=10,
        deep_factor=2,
        width_factor=10,
        act_func=nn.ReLU,
        norm_fun=nn.Identity,
        weight_stardardization=True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=feature_dims[0], kernel_size=3, stride=1, padding=1,
                            weight_standard=False)
        
        self.block1 = WideResNetBlock(in_channels=feature_dims[0], out_channels=feature_dims[0]*width_factor,
                                      stride=1, deep_factor=deep_factor, act_func=act_func, norm_fun=norm_fun,
                                      weight_standardization=weight_stardardization)
        
        self.block2 = WideResNetBlock(in_channels=feature_dims[0]*width_factor, out_channels=feature_dims[1]*width_factor,
                                      stride=2, deep_factor=deep_factor, act_func=act_func, norm_fun=norm_fun,
                                      weight_standardization=weight_stardardization)
        
        self.block3 = WideResNetBlock(in_channels=feature_dims[1]*width_factor, out_channels=feature_dims[2]*width_factor,
                                      stride=2, deep_factor=deep_factor, act_func=act_func, norm_fun=norm_fun,
                                      weight_standardization=weight_stardardization)
        
        self.avg_pool = nn.AvgPool2d(kernel_size=8)
        
        self.classifier_fc = nn.Linear(feature_dims[2]*width_factor, num_classes)
    
    def classify(self, x):
        x = self.forward(x)
        return self.classifier_fc(x)
    
    def energy(self, x):
        x = self.classify(x)
        return torch.logsumexp(x, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avg_pool(x)
        x = x.view(-1, x.shape[1])
        return x
         
        
        
if __name__=="__main__":
    print("Test")
    conv = WideResNet(in_channels=3, deep_factor=8, width_factor=10)
    x = torch.Tensor(1, 3, 32, 32)
    print(conv(x).shape)
    print(sum([p.numel() for p in conv.parameters()]))
