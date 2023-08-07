import pytorch_lightning as pl
import torch


class VanillaTrainer(pl.LightningModule):
    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 config) -> None:
        super(VanillaTrainer, self).__init__()
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.validation_step_outputs = []
    
    def forward(self, x):
        self.model(x)
    
    def calc_loss(self, x, y):
        return self.loss_fn(x, y)

    def calc_accuracy(self, x, y):
        _, pred = x.max(1)
        correct = pred.eq(y).sum().item()
        return torch.Tensor([correct/y.size(0)])

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = self.calc_loss(outputs, labels)
        accuracy = self.calc_accuracy(outputs, labels)
        self.log('loss', loss, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'train_accuracy': accuracy}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = self.calc_loss(outputs, labels)
        accuracy = self.calc_accuracy(outputs, labels)
        return self.validation_step_outputs.append({'valid_loss': loss, 'valid_accuracy': accuracy})
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        mean_loss = torch.stack([x['valid_loss'] for x in outputs]).mean()
        mean_acc = torch.stack([x['valid_accuracy'] for x in outputs]).mean()
        self.log('avg_valid_loss', mean_loss)
        self.log('avg_valid_accuracy', mean_acc)

        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]