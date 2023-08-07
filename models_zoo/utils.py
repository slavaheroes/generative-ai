from wide_resnet import ResNet



MODEL_NAMES = ['resnet', 'wide_resnet', 'convnext']

def make_model(model_name: str):

    assert model_name in MODEL_NAMES, f'{model_name} should be from list {MODEL_NAMES}'
    
    if model_name=='resnet':
        return ResNet(
            N=3,
            num_classes=10
        )
    elif model_name=='wide_resnet':
        raise NotImplementedError
        
    elif model_name=='convnext':
        raise NotImplementedError
    