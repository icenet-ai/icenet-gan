import torch
import torch.nn as nn
import lightning.pytorch as pl

from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torchmetrics import MetricCollection

from .metrics import IceNetAccuracy, SIEError


class LitUNet(pl.LightningModule):
    """
    A LightningModule wrapping the UNet implementation of IceNet.
    """
    def __init__(self,
                 model: nn.Module,
                 criterion: callable,
                 learning_rate: float,
                 enable_leadtime_metrics: bool = True,
                 ):
        """
        Construct a UNet LightningModule.
        Note that we keep hyperparameters separate from dataloaders to prevent data leakage at test time.
        :param model: PyTorch model
        :param criterion: PyTorch loss function for training instantiated with reduction="none"
        :param learning_rate: Float learning rate for our optimiser
        """
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.n_output_classes = model.n_output_classes  # this should be a property of the network

        self.metrics_history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        # Overall metrics
        val_metrics = {
            "val_accuracy": IceNetAccuracy(leadtimes_to_evaluate=list(range(self.model.n_forecast_days))),
            "val_sieerror": SIEError(leadtimes_to_evaluate=list(range(self.model.n_forecast_days))),
        }
        test_metrics = {
            "test_accuracy": IceNetAccuracy(leadtimes_to_evaluate=list(range(self.model.n_forecast_days))),
            "test_sieerror": SIEError(leadtimes_to_evaluate=list(range(self.model.n_forecast_days))),
        }

        # Metrics across each leadtime
        if enable_leadtime_metrics:
            for i in range(self.model.n_forecast_days):
                val_metrics[f"val_accuracy_{i}"] = IceNetAccuracy(leadtimes_to_evaluate=[i])
                val_metrics[f"val_sieerror_{i}"] = SIEError(leadtimes_to_evaluate=[i])

                test_metrics[f"test_accuracy_{i}"] = IceNetAccuracy(leadtimes_to_evaluate=[i])
                test_metrics[f"test_sieerror_{i}"] = SIEError(leadtimes_to_evaluate=[i])

        self.val_metrics = MetricCollection(val_metrics)
        self.test_metrics = MetricCollection(test_metrics)

        # Save input parameters to __init__ (hyperparams) when checkpointing.
        # self.save_hyperparameters(ignore=["model", "criterion"])
        self.save_hyperparameters()


    def forward(self, x):
        """
        Implement forward function.
        :param x: Inputs to model.
        :return: Outputs of model.
        """
        return self.model(x)


    def training_step(self, batch):
        """
        Perform a pass through a batch of training data.
        Apply pixel-weighted loss by manually reducing.
        See e.g. https://discuss.pytorch.org/t/unet-pixel-wise-weighted-loss-function/46689/5.
        :param batch: Batch of input, output, weight triplets
        :param batch_idx: Index of batch
        :return: Loss from this batch of data for use in backprop
        """
        x, y, sample_weight = batch
        outputs = self.model(x)
        # y_hat = torch.sigmoid(outputs)

        loss = self.criterion(outputs, y, sample_weight)

        # This logged result can be accessed later via `self.trainer.callback_metrics("train_loss")`
        # Reference: https://github.com/Lightning-AI/pytorch-lightning/issues/13147#issuecomment-1138975446
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        return {"loss": loss}


    def validation_step(self, batch):
        # x: (b, h, w, channels), y: (b, h, w, n_forecast_days, classes), sample_weight: (b, h, w, n_forecast_days, classes)
        x, y, sample_weight = batch
        outputs = self.model(x)

        # y_hat: (b, h, w, classes, n_forecast_days)
        y_hat = torch.sigmoid(outputs)

        loss = self.criterion(outputs, y, sample_weight)

        self.val_metrics.update(y_hat.squeeze(dim=-2), y.squeeze(dim=-1), sample_weight.squeeze(dim=-1))

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)  # epoch-level loss
        return {"val_loss", loss}


    def test_step(self, batch):
        x, y, sample_weight = batch
        outputs = self.model(x)
        y_hat = torch.sigmoid(outputs)

        loss = self.criterion(outputs, y, sample_weight)

        self.test_metrics.update(y_hat.squeeze(dim=-2), y.squeeze(dim=-1), sample_weight.squeeze(dim=-1))

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)  # epoch-level loss
        return loss


    # def training_epoch_end(self, outputs):
    #     avg_train_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
    #     self.metrics_history["train_loss"].append(avg_train_loss.item())


    # def validation_epoch_end(self, outputs):
    #     avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     avg_val_accuracy = torch.stack([x["val_accuracy"] for x in outputs]).mean()
    #     self.metrics_history["val_loss"].append(avg_train_loss.item())
    #     self.metrics_history["val_accuracy"].append(avg_train_accuracy.item())


    def on_train_epoch_end(self):
        """
        Reference lightning v2.0.0 migration guide:
        https://github.com/Lightning-AI/pytorch-lightning/pull/16520
        """
        # Reference: https://github.com/Lightning-AI/pytorch-lightning/issues/13147#issuecomment-1138975446
        avg_train_loss = self.trainer.callback_metrics["train_loss"]
        self.metrics_history["train_loss"].append(avg_train_loss.item())


    def on_validation_epoch_end(self):
        """
        Reference lightning v2.0.0 migration guide:
        https://github.com/Lightning-AI/pytorch-lightning/pull/16520
        """
        avg_val_loss = self.trainer.callback_metrics["val_loss"]
        self.metrics_history["val_loss"].append(avg_val_loss.item())

        val_accuracy = self.val_metrics["val_accuracy"].compute()
        self.metrics_history["val_accuracy"].append(val_accuracy.item())
        self.log("val_accuracy", val_accuracy, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_metrics.reset()


    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True, sync_dist=True)  # epoch-level metrics
        self.test_metrics.reset()


    def predict_step(self, batch):
        """
        :param batch: Batch of input, output, weight triplets
        :param batch_idx: Index of batch
        :return: Predictions for given input.
        """
        x, y, sample_weight = batch
        y_hat = torch.sigmoid(self.model(x))

        return y_hat


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer
        }


