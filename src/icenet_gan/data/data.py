import logging
import os

import numpy as np
import tensorflow as tf
import torch

from icenet.data.dataset import IceNetDataSet
from icenet.data.datasets.utils import get_decoder
from icenet.data.producers import DataCollection
from torch.utils.data import Dataset, DataLoader

class TFRecordDataset(Dataset):
    def __init__(self, file_paths, decoder):
        self.file_paths = file_paths
        self.decoder = decoder
        self.raw_dataset = tf.data.TFRecordDataset(self.file_paths)
        self.length = sum(1 for _ in self.raw_dataset)

        # self.dataset = self.raw_dataset.map(self.decoder)
        # self.data = list(self.dataset.as_numpy_iterator())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        raw_dataset = self.raw_dataset.skip(idx).take(1)

        # print(type(raw_dataset))
        # x, y, sample_weights = list(raw_dataset.as_numpy_iterator())[0]
        x, y, sample_weights = list(raw_dataset.map(self.decoder))[0]

        # return x, y, sample_weights
        return x.numpy(), y.numpy(), sample_weights.numpy()


        # x, y, sample_weights = self.data[idx]
        # return x, y, sample_weights
        # return torch.tensor(x), torch.tensor(y), torch.tensor(sample_weights)

        # x, y, sample_weights = torch.tensor(sample["x"], dtype=torch.float32)
        # # raw_dataset = tf.data.TFRecordDataset(self.file_paths[idx])
        # # dataset = raw_dataset.map(self.decoder)
        # # x, y, sample_weights = list(self.dataset)
        # dataset = self.dataset.skip(idx).take(1)
        # x, y, sample_weights = self.dataset
        # return x.numpy(), y.numpy(), sample_weights.numpy()

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

class IceNetDataSetTorch(IceNetDataSet):
    def __init__(self, configuration_path, *args, batch_size=4, path=os.path.join(".", "network_datasets"), shuffling=False, **kwargs):
        super().__init__(configuration_path=configuration_path)

        self._config = {}
        self._configuration_path = configuration_path
        self._load_configuration(configuration_path)
        
        # super(IceNetDataSetTorch, self).__init__(*args, identifier=self._config["identifier"], north=bool(self._config["north"]), path=path, south=bool(self._config["south"]), **kwargs)
        DataCollection.__init__(self, *args, identifier=self._config["identifier"], north=bool(self._config["north"]), path=path, south=bool(self._config["south"]), **kwargs)

        self._batch_size = batch_size
        self._dtype = getattr(np, self._config["dtype"])
        self._n_forecast_days = self._config["n_forecast_days"]
        self._num_channels = self._config["num_channels"]
        self._shape = tuple(self._config["shape"])
        self._shuffling = shuffling

        if self._config.get("dataset_path") and os.path.exists(self._config["dataset_path"]):
            hemi = self.hemisphere_str[0]
            if not self.train_fns or not self.val_fns or not self.test_fns:
                self.add_records(self.base_path, hemi)
        else:
            logging.warning("Running in configuration only mode, tfrecords were not generated for this dataset")

    def get_data_loaders(self, ratio=None):
        # train_ds, val_ds, test_ds = self.get_split_datasets(ratio)

        # Wrap TensorFlow datasets with TFRecordDataset
        decoder = get_decoder(self._shape, self._num_channels, self._n_forecast_days, dtype=self._dtype.__name__)

        train_dataset = TFRecordDataset(self.train_fns, decoder)
        val_dataset = TFRecordDataset(self.val_fns, decoder)
        test_dataset = TFRecordDataset(self.test_fns, decoder)

        num_workers = 0
        persistent_workers = True if num_workers else False
        timeout = 30

        # Create PyTorch DataLoader instances
        train_loader = DataLoader(train_dataset,
                                  batch_size=self._batch_size,
                                  shuffle=self._shuffling,
                                  num_workers=num_workers,
                                #   multiprocessing_context="spawn",
                                  persistent_workers=persistent_workers,
                                #   timeout=timeout,
                                  )

        val_loader = DataLoader(val_dataset,
                                batch_size=self._batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                # multiprocessing_context="spawn",
                                persistent_workers=persistent_workers,
                                # timeout=timeout,
                                )

        test_loader = DataLoader(test_dataset,
                                 batch_size=self._batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                #  multiprocessing_context="spawn",
                                 persistent_workers=persistent_workers,
                                #  timeout=timeout,
                                 )

        return train_loader, val_loader, test_loader
