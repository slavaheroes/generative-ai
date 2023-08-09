import random
import numpy as np
import matplotlib.pyplot as plt

import pytorch_lightning as pl
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid

import utils

class LightningTrainer(pl.LightningModule):
    def __init__(
        self,
        model,
        config,
        lr=0.0001,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.config = config
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 50, gamma=0.3)
        
        # channel_size, height, width are for CIFAR10
        self.channels = 3
        self.im_size = 32
        
        self.buffer = torch.FloatTensor(config["buffer_size"], self.channels, self.im_size, self.im_size).cpu()
        self.cls_criterion = torch.nn.CrossEntropyLoss()
        
        self.fid_metric = FrechetInceptionDistance(feature=64)
        self.log_indices = [1] #random.choices(range(100), k=5)
        self.sample_indices = [1] + random.choices(range(2, 100), k=20)
        
        self.validation_outputs = []
        
    def forward(self, x, mode=""):
        if mode=='classify':
            return self.model.classify(x)
        elif mode=='energy':
            return self.model.energy(x)
        return self.model(x)
    
    def _sample_buffer(self, batch_size):
        sample_indices = torch.randint(0, self.config["buffer_size"], (batch_size, ))
        buffer_sample = self.buffer[sample_indices]
        
        random_buffer = torch.FloatTensor(*buffer_sample.shape).uniform_(-1, 1)
        reinit = (torch.rand(batch_size)<self.config['reinit_freq']).float()[:, None, None, None]
                
        samples = (1-reinit)*buffer_sample + reinit*random_buffer
        return samples, sample_indices
    
    def _run_sgld(self, x_hat_0):
        for _ in range(self.config['SGLD_steps']):
            x_hat_0.requires_grad = True
            df_dx = torch.autograd.grad(self.model.energy(x_hat_0).sum(), x_hat_0, retain_graph=True)[0]
            
            x_hat_0.data = x_hat_0 + self.config['SGLD_step_size']*df_dx + self.config['SGLD_noise']*torch.randn_like(x_hat_0)
        
        return x_hat_0
    
    def training_step(self, batch, batch_idx):
        # Algorithm 1: JEM training
        # fixed according to orig implementation
        
        x, y = batch
        # L_clf
        cls_logits = self.model.classify(x)
        L_clf = self.cls_criterion(cls_logits, y)
        
        # sample from B
        x_hat_0, buffer_indices = self._sample_buffer(x.shape[0])
        x_hat_0 = x_hat_0.to(x.device)
        
        # run SGLD
        self.model.eval()
        x_hat_t = self._run_sgld(x_hat_0)
        self.model.train()
        
        # update buffer
        self.buffer[buffer_indices] = x_hat_t.detach().cpu()
        
        # L_gen
        L_gen = self.model.energy(x_hat_t).mean() - self.model.energy(x).mean()
        
        loss = L_clf + L_gen
        
        self.log('L_clf', L_clf, on_epoch=True)
        self.log('L_gen', L_gen, on_epoch=True)
        self.log('loss', loss, on_epoch=True, prog_bar=True)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        cls_logits = self.model.classify(x)
        cls_probas = cls_logits.softmax(1)
        
        preds, confidences = torch.max(cls_probas, dim=1)
        confidences = confidences.cpu().numpy()
        correct = (preds==y).float().cpu().numpy()
        
        self.log('valid_accuracy', correct.sum()/correct.shape[0], on_epoch=True)
        
        self.validation_outputs.append((confidences, correct)) 
        
        if batch_idx in self.sample_indices:
            # sample from B
            torch.set_grad_enabled(True)
            x_hat_0, _ = self._sample_buffer(x.shape[0])
            x_hat_0 = x_hat_0.to(x.device)
            # run SGLD
            fake_images = self._run_sgld(x_hat_0)
            torch.set_grad_enabled(False)
            
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
            
    
    def on_validation_epoch_end(self):
        # calibration plot
        
        confidence = []
        corrects = []
        for conf, corr in self.validation_outputs:
            confidence.extend(conf); corrects.extend(corr)
        
        zipped_corr_conf = np.array(sorted(list(zip(corrects, confidence)), key=lambda x: x[1]))
        corrects = zipped_corr_conf[:, 0]
        confidence = zipped_corr_conf[:, 1]
        
        bucket_accs = utils.get_calibration_bucket(corrects, confidence)
        
        plt.plot([0, 1], [0, 1], linestyle='dashed', color='blue')
        plt.bar(np.arange(len(bucket_accs)), height=bucket_accs, color='red')
        plt.ylim(0.0, 1.0)
        plt.xlim(0.0, 1.0)

        self.logger.experiment.log({"calibration_plot": plt})
        self.validation_outputs.clear()
    
    def configure_optimizers(self):
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]
    
    