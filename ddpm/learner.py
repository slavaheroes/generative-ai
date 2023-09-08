import random
import pytorch_lightning as pl
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class LightningLearner(pl.LightningModule):
    def __init__(
        self,
        unet,
        timesteps,
        beta_start,
        beta_end,
        scheduling,
        lr=0.0005,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['unet'])
        self.unet = unet
        
        self.timesteps = timesteps
        
        # init noise steps
        
        if scheduling=='linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        else:
            raise NotImplementedError
        
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        # optimizers
        
        self.optimizer = torch.optim.AdamW(self.unet.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 25)
        
        # self.fid_metric = FrechetInceptionDistance(feature=64)
        
        self.log_indices = random.choices(range(100), k=10)
        
    def forward(self, x):
        x_hat = self.model(x)
        return x_hat
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.unet(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index==0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    
    @torch.no_grad()
    def p_sample_loop(self, shape, noise=None):
        device = next(self.unet.parameters()).device
        b = shape[0]
        
        if noise is None:
            img = torch.randn(shape, device=device)
        else:
            img = noise
            
        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(img, torch.full((b, ), i, device=device, dtype=torch.long), i)
            
        return img
        
    
    def training_step(self, batch):
        x, _ = batch
        t = torch.randint(0, self.timesteps, (x.shape[0], ), device=x.device).long()
        
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x, t, noise)
        
        noise_pred = self.unet(x_noisy, t)
        
        loss = torch.nn.functional.mse_loss(noise, noise_pred)
        self.log('loss', loss, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        if batch_idx in self.log_indices:
            x, _ = batch
            
            t = torch.ones((x.shape[0], ), device=x.device).long() * (self.timesteps-1)
            noise = torch.randn_like(x)
            
            x_noise = self.q_sample(x, t, noise)
            
            from_given_noise = self.p_sample_loop(x_noise.shape, x_noise)
            from_pure_noise = self.p_sample_loop(x.shape)
            
            # # convert to uint8 type with denormalization
            # self.fid_metric.update((((x+1)/2)*255).type(torch.uint8), real=True)
            # self.fid_metric.update((((from_pure_noise+1)/2)*255).type(torch.uint8), real=False)
            
            # fid_score = self.fid_metric.compute()
            # self.log('FID', fid_score, on_epoch=True)
            
            # self.fid_metric.reset()
                
            # log images
            if batch_idx == self.log_indices[0]:
                grid = make_grid(from_pure_noise, nrow=4, normalize=True)
                self.logger.log_image(key="pure_noise", images=[grid])
                
                grid = make_grid(from_given_noise, nrow=4, normalize=True)
                self.logger.log_image(key="given_noise", images=[grid])
           
                
    def on_validation_epoch_end(self):
        self.log_indices = random.choices(range(100), k=10)
    
    def configure_optimizers(self):
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]
    

    