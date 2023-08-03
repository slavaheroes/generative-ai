import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import torchvision
import torchvision.transforms as transforms

from models import Generator, Discriminator
from trainer import LightningTrainerGAN

BATCH_SIZE = 128
LR = 0.0002
NUM_EPOCHS = 200

if __name__=="__main__":
    logger = WandbLogger(
        entity='slavaheroes',
        project='dcgan'
    )
    
    transform = transforms.Compose(
    [
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomCrop(size=32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = torchvision.datasets.CIFAR10(root='/SSD/slava/', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=8)
    
    testset = torchvision.datasets.CIFAR10(root='/SSD/slava/', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=200,
                                         shuffle=False, num_workers=8)
    
    pl_model = LightningTrainerGAN(
        generator=Generator(),
        discriminator=Discriminator(),
        lr=LR
    )
    
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[1],
        max_epochs=NUM_EPOCHS,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=5,
        logger=logger,
        log_every_n_steps=2,
        # precision='16-mixed'
    )
    
    trainer.fit(pl_model, trainloader, testloader)
    