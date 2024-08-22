import logging
import os

import numpy as np
import tensorflow as tf

from icenet.data.dataset import IceNetDataSet
from icenet.data.datasets.utils import get_decoder
from icenet.data.producers import DataCollection
from torch.utils.data import Dataset, DataLoader

class TFRecordDataset(Dataset):
    def __init__(self, file_paths, decoder):
        self.file_paths = file_paths
        self.decoder = decoder

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        raw_dataset = tf.data.TFRecordDataset(self.file_paths[idx])
        dataset = raw_dataset.map(self.decoder)
        x, y, sample_weights = list(dataset)[0]
        # dataset = dataset.skip(idx).take(1)
        # print(list(dataset))
        # return list(dataset)
        return x.numpy(), y.numpy(), sample_weights.numpy()

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

class IceNetDataSetTorch(IceNetDataSet):
    def __init__(self, configuration_path, *args, batch_size=4, path=os.path.join(".", "network_datasets"), shuffling=False, **kwargs):
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
        train_ds, val_ds, test_ds = self.get_split_datasets(ratio)

        # Wrap TensorFlow datasets with TFRecordDataset
        decoder = get_decoder(self._shape, self._num_channels, self._n_forecast_days, dtype=self._dtype.__name__)

        train_dataset = TFRecordDataset(self.train_fns, decoder)
        val_dataset = TFRecordDataset(self.val_fns, decoder)
        test_dataset = TFRecordDataset(self.test_fns, decoder)

        # Create PyTorch DataLoader instances
        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=self._shuffling,)
        val_loader = DataLoader(val_dataset, batch_size=self._batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False)

        return train_loader, val_loader, test_loader