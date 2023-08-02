import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

import torchvision
import torchvision.transforms as transforms

from model import VAE
from trainer import LightningTrainer

BATCH_SIZE = 128
LR = 0.0005
NUM_EPOCHS = 200

if __name__=="__main__":
    logger = WandbLogger(
        entity='slavaheroes',
        project='vae',
        name='cnn_cifar10'
    )
    
    transform = transforms.Compose(
    [
        # transforms.Pad(padding=4, fill=0),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomCrop(size=32, padding=4),
        transforms.ToTensor()
    ])
    
    dataset = torchvision.datasets.CIFAR10(root='/SSD/slava/', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=8)
    
    testset = torchvision.datasets.CIFAR10(root='/SSD/slava/', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=200,
                                         shuffle=False, num_workers=8)
    
    model = VAE()
    pl_model = LightningTrainer(model=model, lr=LR)
    
    callbacks=[
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        max_epochs=NUM_EPOCHS,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=5,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=2,
        
        # precision='16-mixed'
    )
    
    trainer.fit(pl_model, trainloader, testloader)
    