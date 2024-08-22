"""Main module."""
import logging
import os

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.utils.data as data

from icenet.data.dataset import IceNetDataSetPyTorch
from icenet.model.cli import TrainingArgParser
from icenet.model.networks.base import BaseNetwork
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from .data.data import IceNetDataSetTorch
from .models.models import unet_batchnorm
from .models.losses import WeightedMSELoss

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

        save_top_k = 1 if save else 0

        checkpoint_callback = ModelCheckpoint(monitor="val_accuracy",
                                              mode="max",
                                              save_top_k=save_top_k,
                                              dirpath=self.model_path,
                                              )

        logging.info("Saving model to: {}".format(self._model_path))
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


def get_datasets(args):
    # TODO: this should come from a factory in the future - not the only place
    #  that merged datasets are going to be available

    dataset_filenames = [
        el if str(el).split(".")[-1] == "json" else "dataset_config.{}.json".format(el)
        for el in [args.dataset, *args.additional]
    ]

    # if len(args.additional) == 0:
    #     dataset = IceNetDataSetPyTorch(dataset_filenames[0],
    #                                     batch_size=args.batch_size,
    #                                     shuffling=args.shuffle_train)
    # else:
    #     dataset = MergedIceNetDataSet(dataset_filenames,
    #                                   batch_size=args.batch_size,
    #                                   shuffling=args.shuffle_train)

    dataset = IceNetDataSetTorch(dataset_filenames[0],
                                batch_size=args.batch_size,
                                shuffling=False
                                )
    return dataset


def pytorch_main():
    args = TrainingArgParser().add_unet().parse_args()
    dataset = get_datasets(args)
    network = PytorchNetwork(dataset,
                                args.run_name,
                                checkpoint_mode=args.checkpoint_mode,
                                checkpoint_monitor=args.checkpoint_monitor,
                                early_stopping_patience=args.early_stopping,
                                lr_decay=(
                                    args.lr_10e_decay_fac,
                                    args.lr_decay_start,
                                    args.lr_decay_end,
                                ),
                                pre_load_path=args.preload,
                                seed=args.seed,
                                verbose=args.verbose)
    execute_pytorch_training(args, dataset, network)


def execute_pytorch_training(args, dataset, network,
                        save=True,
                        evaluate=True):
    # There is a better way of doing this by passing off to a dynamic factory
    # for other integrations, but for the moment I have no shame
    using_wandb = False
    run = None

    # # TODO: move to overridden implementation - decorator?
    # if not args.no_wandb:
    #     from icenet.model.handlers.wandb import init_wandb, finalise_wandb
    #     run, callback = init_wandb(args)

    #     if callback is not None:
    #         network.add_callback(callback)
    #         using_wandb = True

    input_shape = (*dataset.shape, dataset.num_channels)
    ratio = args.ratio if args.ratio else 1.0
    # Do not yet support ratio != 1.0 with pytorch
    train_dataloader, validation_dataloader, _ = dataset.get_data_loaders(ratio=1.0)
    # train_ds, val_ds, _ = dataset.get_split_datasets(ratio=ratio)

    network.train(
        args.epochs,
        unet_batchnorm,
        train_dataloader,
        model_creator_kwargs=dict(
            input_shape=input_shape,
            # loss=losses.WeightedMSE(),
            loss=WeightedMSELoss(reduction="none"),
            # metrics=[
            #     metrics.WeightedBinaryAccuracy(),
            #     metrics.WeightedMAE(),
            #     metrics.WeightedRMSE(),
            #     losses.WeightedMSE()
            # ],
            learning_rate=args.lr,
            filter_size=args.filter_size,
            n_filters_factor=args.n_filters_factor,
            n_forecast_days=dataset.n_forecast_days,
        ),
        save=save,
        validation_dataloader=validation_dataloader,
    )

    # if evaluate:
    #     results, metric_names, leads = \
    #         evaluate_model(network.model_path,
    #                        dataset,
    #                        dataset_ratio=args.ratio)

    #     if using_wandb:
    #         finalise_wandb(run, results, metric_names, leads)
