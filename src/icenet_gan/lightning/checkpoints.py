import logging

import torch

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

class ModelCheckpointOnImprovement(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_score = None

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        monitor_candidates = self._monitor_candidates(trainer)
        current_score = monitor_candidates[self.monitor]

        monitor_op = {"min": torch.lt, "max": torch.gt}[self.mode]
        logging.debug("Monitor candidates:", self.mode, monitor_candidates)
        # Check if metric's best score has improved.
        if self.best_score is None or monitor_op(current_score, self.best_score):
            logging.info(f"Checkpoint saved at epoch {trainer.current_epoch} with {self.monitor}: {current_score:.4f}")
            self.best_score = current_score

            # Only save checkpoint if score has improved
            super().on_train_epoch_end(trainer, pl_module)
        else:
            logging.info(f"No improvement in {self.monitor} at epoch {trainer.current_epoch}: {current_score:.4f} (Best: {self.best_score:.4f})")
