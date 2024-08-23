"""Main module."""
import logging
import os

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.utils.data as data


from icenet.model.networks.base import BaseNetwork
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from ...lightning.checkpoints import ModelCheckpointOnImprovement

class PytorchNetwork(BaseNetwork):
    def __init__(self,
                 *args,
                 checkpoint_mode: str = "min",
                 checkpoint_monitor: str = None,
                 early_stopping_patience: int = 0,
                 lr_decay: tuple = (1.0, 0, 0),
                 pre_load_path: str = None,
                 tensorboard_logdir: str = None,
                 verbose: bool = False,
                 **kwargs):
        self._checkpoint_mode = checkpoint_mode
        self._checkpoint_monitor = checkpoint_monitor
        self._early_stopping_patience = early_stopping_patience
        self._lr_decay = lr_decay
        self._tensorboard_logdir = tensorboard_logdir

        super().__init__(*args, **kwargs)

        self._weights_path = os.path.join(
            self.network_folder, "{}.network_{}.{}.h5".format(
                self.run_name, self.dataset.identifier, self.seed))

        if pre_load_path is not None and not os.path.exists(pre_load_path):
            raise RuntimeError("{} is not available, so you cannot preload the "
                               "network with it!".format(pre_load_path))
        self._pre_load_path = pre_load_path

        self._verbose = verbose

        torch.set_float32_matmul_precision("medium")


    def _attempt_seed_setup(self):
        super()._attempt_seed_setup()
        pl.seed_everything(self._seed)


    def train(self,
              epochs: int,
              model_creator: callable,
              train_dataloader: object,
              model_creator_kwargs: dict = None,
              save: bool = True,
              validation_dataloader: object = None):

        history_path = os.path.join(self.network_folder,
                                    "{}_{}_history.json".format(
                                        self.run_name, self.seed))

        logger_name = f"{self.run_name}_{self.seed}"

        lit_module = model_creator(**model_creator_kwargs)
        logger = CSVLogger("logs", name=logger_name)

        print(lit_module.model)

        # set up trainer configuration
        trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            log_every_n_steps=5,
            max_epochs=epochs,
            num_sanity_val_steps=0,
            enable_checkpointing=False,
            logger=logger,
            fast_dev_run=False, # Runs single batch through train and validation
                                #    when running trainer.test()
                                # Note: Cannot use with automatic best checkpointing
        )

        # save_top_k = 1 if save else 0

        if save:
            checkpoint_filename = "checkpoint.{}.network_{}.{}.".format(
                    self.run_name, self.dataset.identifier, self.seed) + \
                    "{epoch:03d}"

            checkpoint_callback = ModelCheckpointOnImprovement(monitor="val_accuracy",
                                                mode=self._checkpoint_mode,
                                                save_top_k=-1,
                                                # every_n_epochs=1,
                                                filename=checkpoint_filename,
                                                # Prevents "epoch=001" in filename output
                                                auto_insert_metric_name=False,
                                                # dirpath=self._weights_path,
                                                dirpath=self.network_folder,
                                                save_weights_only=True,
                                                )

            logging.info("Saving model & network to: {}".format(self._weights_path))
            trainer.callbacks.append(checkpoint_callback)

        # train model
        # print(f"Training {len(train_dataset)} examples / {len(train_dataloader)} batches (batch size {batch_size}).")
        # print(f"Validating {len(validation_dataset)} examples / {len(val_dataloader)} batches (batch size {batch_size}).")
        if self._pre_load_path and os.path.exists(self._pre_load_path):
            logging.warning("Automagically loading network weights from {}".format(
                self._pre_load_path))

        trainer.fit(
            lit_module,
            train_dataloader,
            validation_dataloader,
            ckpt_path=None,
        )

        with open(history_path, 'w') as fh:
            logging.info(f"Saving metrics history to: {history_path}")
            print(lit_module.metrics_history)
            pd.DataFrame(lit_module.metrics_history).to_json(fh)

        # # TODO: consider using .keras format throughout
        # # TODO: need to consider pre_load / create and save functionality for checkpoint recovery
        # if self._pre_load_path and os.path.exists(self._pre_load_path):
        #     logging.warning("Automagically loading network weights from {}".format(
        #         self._pre_load_path))
        #     network.load_weights(self._pre_load_path)

        # network.summary()

        # if save:
        #     logging.info("Saving network to: {}".format(self._weights_path))
        #     network.save_weights(self._weights_path)
        #     logging.info("Saving model to: {}".format(self.model_path))
        #     save_model(network, self.model_path)
        ## To save model history, should define a callback to process the logging output.
        #     with open(history_path, 'w') as fh:
        #         pd.DataFrame(model_history.history).to_json(fh)

        return trainer, checkpoint_callback
