import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

import torch
import torchvision
import torchvision.transforms as transforms

import utils
from vanilla_learner import VanillaTrainer


train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(size=32, padding=4),
        transforms.ToTensor(), # between 0 and 1
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # between -1 and 1
    ])

test_transform = transforms.Compose(
    [
        transforms.ToTensor(), # between 0 and 1
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # between -1 and 1
    ])


if __name__=="__main__":
    '''
    The default training setting is adapted from ViT
    To change the parameters of a model, do it manually in utils.py
    '''
    parser = argparse.ArgumentParser(description="Pass initial LR, BatchSize, weight decay, epochs, model name")
    parser.add_argument('--lr', type=float, default=4e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model_name', type=str)
    args = parser.parse_args()
    
    lr = args.lr
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    epochs = args.epochs
    model_name = args.model_name
    
    model = utils.make_model(model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    
    logger = WandbLogger(
        entity='slavaheroes',
        project='classification_models',
        name=f'{model_name}'
    )
    
    trainset = torchvision.datasets.CIFAR10(root='/SSD/slava/', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=16)
    
    testset = torchvision.datasets.CIFAR10(root='/SSD/slava/', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=16)
    
    pl_model = VanillaTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=args
    )
    
    callbacks = [LearningRateMonitor(logging_interval='epoch')]
    
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[1],
        max_epochs=epochs,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=5,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=2,
        precision='16-mixed'
    )
    
    trainer.fit(pl_model, trainloader, testloader)