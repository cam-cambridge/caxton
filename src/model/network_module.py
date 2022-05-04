import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.residual_attention_network import (
    ResidualAttentionModel_56 as ResidualAttentionModel,
)
import pytorch_lightning as pl
from datetime import datetime
import pandas as pd
import os

class ParametersClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        lr=1e-3,
        transfer=False,
        trainable_layers=1,
        gpus=1,
        retrieve_layers=False,
        retrieve_masks=False,
        test_overwrite_filename=False,
    ):
        super().__init__()
        self.lr = lr
        self.__dict__.update(locals())
        self.attention_model = ResidualAttentionModel(
            retrieve_layers=retrieve_layers, retrieve_masks=retrieve_masks
        )
        num_ftrs = self.attention_model.fc.in_features
        self.attention_model.fc = nn.Identity()
        self.fc1 = nn.Linear(num_ftrs, num_classes)
        self.fc2 = nn.Linear(num_ftrs, num_classes)
        self.fc3 = nn.Linear(num_ftrs, num_classes)
        self.fc4 = nn.Linear(num_ftrs, num_classes)

        if transfer:
            for child in list(self.attention_model.children())[:-trainable_layers]:
                for param in child.parameters():
                    param.requires_grad = False
        self.save_hyperparameters()

        self.train_acc = pl.metrics.Accuracy()
        self.train_acc0 = pl.metrics.Accuracy()
        self.train_acc1 = pl.metrics.Accuracy()
        self.train_acc2 = pl.metrics.Accuracy()
        self.train_acc3 = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.val_acc0 = pl.metrics.Accuracy()
        self.val_acc1 = pl.metrics.Accuracy()
        self.val_acc2 = pl.metrics.Accuracy()
        self.val_acc3 = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

        self.name = "ResidualAttentionClassifier"
        self.retrieve_layers = retrieve_layers
        self.retrieve_masks = retrieve_masks
        self.gpus = gpus
        self.sync_dist = True if self.gpus > 1 else False
        self.test_overwrite_filename = test_overwrite_filename

    def forward(self, X):
        X = self.attention_model(X)
        if self.retrieve_layers or self.retrieve_masks:
            out1 = self.fc1(X[0])
            out2 = self.fc2(X[0])
            out3 = self.fc3(X[0])
            out4 = self.fc4(X[0])
            return (out1, out2, out3, out4), X
        out1 = self.fc1(X)
        out2 = self.fc2(X)
        out3 = self.fc3(X)
        out4 = self.fc4(X)
        return (out1, out2, out3, out4)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=3, threshold=0.01
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hats = self.forward(x)
        y_hat0, y_hat1, y_hat2, y_hat3 = y_hats
        y = y.t()

        _, preds0 = torch.max(y_hat0, 1)
        loss0 = F.cross_entropy(y_hat0, y[0])

        _, preds1 = torch.max(y_hat1, 1)
        loss1 = F.cross_entropy(y_hat1, y[1])

        _, preds2 = torch.max(y_hat2, 1)
        loss2 = F.cross_entropy(y_hat2, y[2])

        _, preds3 = torch.max(y_hat3, 1)
        loss3 = F.cross_entropy(y_hat3, y[3])

        loss = loss0 + loss1 + loss2 + loss3
        preds = torch.stack((preds0, preds1, preds2, preds3))

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "train_loss0",
            loss0,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "train_loss1",
            loss1,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "train_loss2",
            loss2,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "train_loss3",
            loss3,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )

        self.train_acc(preds, y)
        self.train_acc0(preds0, y[0])
        self.train_acc1(preds1, y[1])
        self.train_acc2(preds2, y[2])
        self.train_acc3(preds3, y[3])

        self.log(
            "train_acc",
            self.train_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "train_acc0",
            self.train_acc0,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "train_acc1",
            self.train_acc1,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "train_acc2",
            self.train_acc2,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "train_acc3",
            self.train_acc3,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )

        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hats = self.forward(x)
        y_hat0, y_hat1, y_hat2, y_hat3 = y_hats
        y = y.t()

        _, preds0 = torch.max(y_hat0, 1)
        loss0 = F.cross_entropy(y_hat0, y[0])

        _, preds1 = torch.max(y_hat1, 1)
        loss1 = F.cross_entropy(y_hat1, y[1])

        _, preds2 = torch.max(y_hat2, 1)
        loss2 = F.cross_entropy(y_hat2, y[2])

        _, preds3 = torch.max(y_hat3, 1)
        loss3 = F.cross_entropy(y_hat3, y[3])

        loss = loss0 + loss1 + loss2 + loss3
        preds = torch.stack((preds0, preds1, preds2, preds3))

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "val_loss0",
            loss0,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "val_loss1",
            loss1,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "val_loss2",
            loss2,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "val_loss3",
            loss3,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )

        self.val_acc(preds, y)
        self.val_acc0(preds0, y[0])
        self.val_acc1(preds1, y[1])
        self.val_acc2(preds2, y[2])
        self.val_acc3(preds3, y[3])

        self.log(
            "val_acc",
            self.val_acc,
            prog_bar=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "val_acc0",
            self.val_acc0,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "val_acc1",
            self.val_acc1,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "val_acc2",
            self.val_acc2,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "val_acc3",
            self.val_acc3,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hats = self.forward(x)
        y_hat0, y_hat1, y_hat2, y_hat3 = y_hats
        y = y.t()

        _, preds0 = torch.max(y_hat0, 1)
        loss0 = F.cross_entropy(y_hat0, y[0])

        _, preds1 = torch.max(y_hat1, 1)
        loss1 = F.cross_entropy(y_hat1, y[1])

        _, preds2 = torch.max(y_hat2, 1)
        loss2 = F.cross_entropy(y_hat2, y[2])

        _, preds3 = torch.max(y_hat3, 1)
        loss3 = F.cross_entropy(y_hat3, y[3])

        loss = loss0 + loss1 + loss2 + loss3

        self.log("test_loss0", loss0)
        self.log("test_loss1", loss1)
        self.log("test_loss2", loss2)
        self.log("test_loss3", loss3)

        preds = torch.stack((preds0, preds1, preds2, preds3))
        self.log(
            "test_loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.test_acc(preds, y)
        self.log(
            "test_acc",
            self.test_acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        return {"loss": loss, "preds": preds, "targets": y}

    def test_epoch_end(self, outputs):
        preds = [output["preds"] for output in outputs]
        targets = [output["targets"] for output in outputs]

        preds = torch.cat(preds, dim=1)
        targets = torch.cat(targets, dim=1)

        os.makedirs("test/", exist_ok=True)
        if self.test_overwrite_filename:
            torch.save(preds, "test/preds_test.pt")
            torch.save(targets, "test/targets_test.pt")
        else:
            date_string = datetime.now().strftime("%H-%M_%d-%m-%y")
            torch.save(preds, "test/preds_{}.pt".format(date_string))
            torch.save(targets, "test/targets_{}.pt".format(date_string))
