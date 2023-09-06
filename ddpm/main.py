import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torchvision
import torchvision.transforms as transforms

from learner import LightningLearner
from model import UNet

BATCH_SIZE = 50
LR = 0.0002
NUM_EPOCHS = 100

config = {
    "timesteps": 200,
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "schedule_type": "linear"
}


if __name__=="__main__":
    logger = WandbLogger(
        entity='slavaheroes',
        project='DDPM'
    )
    
    transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
        # lambda x: x + 0.03 * torch.randn_like(x)
    ])
    
    dataset = torchvision.datasets.CIFAR10(root='/SSD/slava/', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=8)
    
    testset = torchvision.datasets.CIFAR10(root='/SSD/slava/', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=40,
                                         shuffle=False, num_workers=8)
    
    model = UNet(dim=64, downsample_w_stride=False, upsample_w_transpose=False, width_factor=(1, 2, 4, 8),
                 in_channels=3)
    
    pl_model = LightningLearner(unet=model,
                                timesteps=config["timesteps"],
                                beta_start=config["beta_start"],
                                beta_end=config["beta_end"],
                                scheduling=config["schedule_type"],
                                lr=LR)
    
    callbacks=[
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            save_top_k=-1,
            dirpath='/SSD/slava/generative_ai/DDPM'
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
    