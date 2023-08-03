import random
import pytorch_lightning as pl
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid

class LightningTrainerGAN(pl.LightningModule):
    def __init__(
        self,
        generator,
        discriminator,
        lr=0.0005,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['generator', 'discriminator'])
        self.automatic_optimization = False
        
        self.generator = generator
        self.discriminator = discriminator
        
        self.g_opt = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
        self.d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
        
        self.fid_metric = FrechetInceptionDistance(feature=64)
        
        self.log_indices = [1] #random.choices(range(100), k=5)
        
    def forward(self, x):
        return self.generator(x)
    
    def training_step(self, batch, batch_idx):
        real_images, _ = batch
        optimizer_g, optimizer_d = self.optimizers()
        
        z = torch.randn(real_images.shape[0], 100).type_as(real_images)
        real_labels = torch.ones(real_images.shape[0], 1).type_as(real_images)
        fake_labels = torch.zeros(real_images.shape[0], 1).type_as(real_images)
        
        # Train D
        
        self.toggle_optimizer(optimizer_d)
        optimizer_d.zero_grad()
        
        real_preds = self.discriminator(real_images)
        real_loss = torch.nn.functional.binary_cross_entropy(real_preds, real_labels)
        
        fake_images = self.generator(z)
        fake_preds = self.discriminator(fake_images)
        fake_loss = torch.nn.functional.binary_cross_entropy(fake_preds, fake_labels)
        
        d_loss = real_loss + fake_loss
        
        self.log("d_loss", d_loss, prog_bar=True, on_epoch=True)
        self.manual_backward(d_loss)
        
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)
        
        # Train G
        
        z = torch.randn(real_images.shape[0], 100).type_as(real_images)
        self.toggle_optimizer(optimizer_g)
        optimizer_g.zero_grad()
        
        fake_images = self.generator(z)
        fake_preds = self.discriminator(fake_images)
        g_loss = torch.nn.functional.binary_cross_entropy(fake_preds, real_labels)
        
        self.log("g_loss", g_loss, prog_bar=True, on_epoch=True)
        self.manual_backward(g_loss)
        
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)
        
        
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        z = torch.randn(x.shape[0], 100).type_as(x)
        fake_images =  self(z)
        
        # convert to uint8 type with denormalization
        self.fid_metric.update((((x+1)/2)*255).type(torch.uint8), real=True)
        self.fid_metric.update((((fake_images+1)/2)*255).type(torch.uint8), real=False)
        
        fid_score = self.fid_metric.compute()
        self.log('FID', fid_score, on_epoch=True)
        
        self.fid_metric.reset()
        
        # log image
        if batch_idx in self.log_indices:
            grid = make_grid(fake_images, nrow=10, normalize=True)
            self.logger.log_image(key="fake_images", images=[grid])
            
            grid = make_grid(x, nrow=10, normalize=True)
            self.logger.log_image
            
    
    def configure_optimizers(self):
        return [self.g_opt, self.d_opt], []
    