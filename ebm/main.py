import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torchvision
import torchvision.transforms as transforms

from model import WideResNet
from trainer import LightningTrainer

BATCH_SIZE = 128
LR = 0.0001
NUM_EPOCHS = 100

config = {
    "SGLD_steps": 20,
    "buffer_size": 10000,
    "reinit_freq": 0.05,
    "SGLD_step_size": 1.,
    "SGLD_noise": 0.01
}

CONTINUE_PATH = '/SSD/slava/generative_ai/EBM/epoch=3-step=1564-v1.ckpt'


if __name__=="__main__":
    logger = WandbLogger(
        entity='slavaheroes',
        project='EBM'
    )
    
    transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        lambda x: x + 0.03 * torch.randn_like(x)
    ])
    
    dataset = torchvision.datasets.CIFAR10(root='/SSD/slava/', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=8)
    
    testset = torchvision.datasets.CIFAR10(root='/SSD/slava/', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=200,
                                         shuffle=False, num_workers=8)
    
    model = WideResNet(deep_factor=8, width_factor=10, num_classes=10)
    
    if len(CONTINUE_PATH)==0:
        pl_model = LightningTrainer(model=model, config=config, lr=LR)
    else:
        pl_model = LightningTrainer.load_from_checkpoint(CONTINUE_PATH, model=model, config=config, lr=LR)
    
    callbacks=[
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            save_top_k=1,
            monitor='valid_accuracy',
            mode='max',
            dirpath='/SSD/slava/generative_ai/EBM'
        )
    ]
    
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[1],
        max_epochs=NUM_EPOCHS,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=2,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=2,
        # precision='16-mixed',
        # limit_train_batches=5,
        # limit_val_batches=5,
    )
    
    trainer.fit(pl_model, trainloader, testloader)
    