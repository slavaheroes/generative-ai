import random
import pytorch_lightning as pl
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid

class LightningTrainer(pl.LightningModule):
    def __init__(
        self,
        model,
        lr=0.0005,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 25)
        
        self.fid_metric = FrechetInceptionDistance(feature=64)
        
        self.log_indices = [1] #random.choices(range(100), k=5)
        
    def forward(self, x):
        x_hat, loss = self.model(x)
        return x_hat, loss
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        _, loss = self(x)
        self.log('loss', loss, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        batch_size = x.shape[0]
        fake_images =  self.model.sample(batch_size)
        
        # convert to uint8 type with denormalization
        self.fid_metric.update((((x+1)/2)*255).type(torch.uint8), real=True)
        self.fid_metric.update((((fake_images+1)/2)*255).type(torch.uint8), real=False)
        
        fid_score = self.fid_metric.compute()
        self.log('FID', fid_score, on_epoch=True)
        
        self.fid_metric.reset()
        
        # log image
        if batch_idx in self.log_indices:
            grid = make_grid(fake_images, nrow=10, normalize=True)
            self.logger.log_image(key="fake_images_noi", images=[grid])
            
            x_hat, _ = self(x)
            grid = make_grid(x_hat, nrow=10, normalize=True)
            self.logger.log_image(key="fake_images", images=[grid])
    
    def configure_optimizers(self):
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]
    

    