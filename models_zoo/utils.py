from models.resnet import ResNet
from models.wide_resnet import WideResNet
from models.convnext import ConvNeXt

MODEL_NAMES = ['resnet', 'wide_resnet', 'convnext']

def make_model(model_name: str):

    assert model_name in MODEL_NAMES, f'{model_name} should be from list {MODEL_NAMES}'
    
    if model_name=='resnet':
        return ResNet(
            N=3,
            num_classes=10
        )
    elif model_name=='wide_resnet':
        return WideResNet(
            in_channels=3,
            num_classes=10,
            deep_factor=8,
            width_factor=10,
            weight_stardardization=True
        )
        
    elif model_name=='convnext':
        return ConvNeXt(
            in_channels=3,
            num_classes=10,
            channels=[96, 192, 384, 768],
            num_blocks=[3, 3, 27, 3],
            drop_path_rate=0.2
        )
    