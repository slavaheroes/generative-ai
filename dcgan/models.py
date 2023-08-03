import torch.nn as nn
import torch

class Generator(nn.Module):
    '''
    Generator class.
    Forward function takes a noise tensor of size Bx100
    and returns Bx3x32x32
    '''
    def __init__(self, out_channels=3, noise_size=100):
        super().__init__()
        self.model = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=noise_size, out_channels=512, stride=1, kernel_size=4, padding=0, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, stride=2, kernel_size=4, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, stride=2, kernel_size=4, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=out_channels, stride=2, kernel_size=4, padding=1, bias=False),
            nn.Tanh()
        ])
        
    
    def forward(self, x):
        x = x.reshape((x.shape[0], x.shape[1], 1, 1))
        return self.model(x)

  
class Discriminator(nn.Module):
    '''
    Discriminator class.
    Forward function takes a tensor of size BxCx32x32
    and return Bx1
    '''
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(*[
            nn.Conv2d(in_channels=in_channels, out_channels=128, stride=2, kernel_size=5, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=128, out_channels=256, stride=2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=256, out_channels=512, stride=2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=512, out_channels=1, stride=4, kernel_size=5, padding=1, bias=False),
            nn.Sigmoid()
        ])
    
    def forward(self, x):
        x = self.model(x)
        return x.reshape((x.shape[0], 1))
    
if __name__=="__main__":
    x = torch.randn(32, 100)
    gen = Generator()
    x_fake = gen(x)
    print(x_fake.shape)
    
    dis = Discriminator()
    y = dis(x_fake)
    print(y.shape)