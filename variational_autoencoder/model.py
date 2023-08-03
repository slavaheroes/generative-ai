import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=in_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
        )
        
        self.mu = nn.Linear(128*4*4, latent_dim, bias=False)
        self.log_var = nn.Linear(128*4*4, latent_dim, bias=False)
        self.projector = nn.Linear(latent_dim, 128*4*4, bias=False)
        
        self.latent_dim = latent_dim
        self.last_channel_dim = 128
        
        
    def forward(self, x):
        
        x_encoded = self.encoder(x)

        mu, log_var = self.mu(x_encoded), self.log_var(x_encoded)
        noise = torch.randn_like(log_var).to(log_var.device)
        
        latent = mu + torch.exp(0.5*log_var)*noise
        proj_latent = self.projector(latent).view(-1, self.last_channel_dim, 4, 4)
        
        x_hat = self.decoder(proj_latent)
        # print(x_hat.shape)
        # compute loss
        loss = self.compute_loss(x, x_hat, mu, log_var)
        
        return x_hat, loss
    
    def compute_loss(self, x, x_hat, mean, log_var):
        recon_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        kl_loss = -0.5*torch.sum(1+log_var-torch.pow(mean, 2)-torch.exp(log_var))
        
        return recon_loss+kl_loss
        
        
    def sample(self, batch_size):
        noise = torch.randn(batch_size, self.latent_dim).to(next(self.parameters()).device)
        proj_latent = self.projector(noise).view(-1, self.last_channel_dim, 4, 4)
        fake_images = self.decoder(proj_latent)
        return fake_images
        

if __name__=="__main__":
    x = torch.Tensor(1, 3, 32, 32)
    model = VAE()
    y, loss = model(x)
    print(y.shape, loss)
    
    sample = model.sample(batch_size=10)
    print(sample.shape)
