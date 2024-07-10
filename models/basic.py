import pytorch_lightning as pl


class UNetModel(pl.LightningModule):
    def __init__(self, net, loss, learning_rate, optimizer, dwt):
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.loss = loss
        self.optimizer_class = optimizer
        self.dwt = dwt

    def load_batch(self, batch):
        input, target = batch
        target = self.dwt(target)
        pred = self.net(input)
        return pred, target

    def training_step(self, batch, batch_idx):
        pred, target = self.load_batch(batch)
        loss = self.loss(pred, target)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, target = self.load_batch(batch)
        loss = self.loss(pred, target)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        pred, target = self.load_batch(batch)
        loss = self.loss(pred, target)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer
