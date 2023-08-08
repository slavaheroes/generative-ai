import torch.nn as nn

class ResNetBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=3, padding=1, stride=stride, bias=False)
    self.bn1 = nn.BatchNorm2d(num_features=out_channels)
    self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                         kernel_size=3, padding=1, stride=1, bias=False)
    self.bn2 = nn.BatchNorm2d(num_features=out_channels)
    self.relu = nn.ReLU()

    self.use_padding = (in_channels != out_channels)


  def forward(self, x):
    x1 = self.relu(self.bn1(self.conv1(x)))
    x2 = self.bn2(self.conv2(x1))

    if self.use_padding:
      # dimensions are not matched hence we need to use shortcut
      # tensor: BxCxHxW
      padding_shape = []
              
      for i in range(len(x.shape)-1, -1, -1):
        padding_shape.append((x2.shape[i] - x.shape[i])//2)
        padding_shape.append((x2.shape[i] - x.shape[i])//2)

      x = nn.functional.pad(x, padding_shape, "constant", 0)

    return self.relu(x2 + x)

class ResNetLayer(nn.Module):
  def __init__(self, in_channels, out_channels, first_stride, N=3):
    super().__init__()
    layers = [
        ResNetBlock(in_channels=in_channels, out_channels=out_channels, stride=first_stride)
    ]

    for _ in range(N-1):
      layers.append(
          ResNetBlock(in_channels=out_channels, out_channels=out_channels, stride=1)
      )
    
    self.layer = nn.Sequential(*layers)
      
  def forward(self, x):
    return self.layer(x)

class ResNet(nn.Module):
    def __init__(self, N=3, num_classes=10):
        super().__init__()
        self.N = N
        self.num_classes = num_classes

        # First conv layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.relu = nn.ReLU()

        # 32x32 map size layers
        self.layer32 = ResNetLayer(in_channels=16, out_channels=16, first_stride=1, N=N)

        # 16x16 map size layers
        self.layer16 = ResNetLayer(in_channels=16, out_channels=32, first_stride=2, N=N)

        # 8x8 map size layers
        self.layer8 = ResNetLayer(in_channels=32, out_channels=64, first_stride=2, N=N)

        # avg pool and fc layers
        self.avg_pool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
      x = self.relu(self.bn1(self.conv1(x)))

      x = self.layer32(x)

      x = self.layer16(x)

      x = self.layer8(x)

      x = self.avg_pool(x)
      x = self.fc(x.squeeze())
      
      return x