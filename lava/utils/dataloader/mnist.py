# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import os
import numpy as np


class MnistDataset:
    def __init__(self, data_path=os.path.join(os.path.dirname(__file__),
                                              'mnist.npy')):
        """data_path (str): Path to mnist.npy file containing the MNIST
        dataset"""
        if not os.path.exists(data_path):
            # Download MNIST from internet and convert it to .npy
            os.makedirs(os.path.join(os.path.dirname(__file__), 'temp'),
                        exist_ok=False)
            MnistDataset. \
                download_mnist(path=os.path.join(
                    os.path.dirname(__file__),
                    'temp'
                )
                )
            # GUnzip, Parse and save MNIST data as .npy
            MnistDataset.decompress_convert_save(
                download_path=os.path.join(os.path.dirname(__file__), 'temp'),
                save_path=os.path.dirname(data_path))
        self.data = np.load(data_path, allow_pickle=True)

    # ToDo: Populate this method with a proper wget download from MNIST website
    @staticmethod
    def download_mnist(path=os.path.join(os.path.dirname(__file__), 'temp')):
        pass

    # ToDo: Populate this method with proper code to decompress, parse,
    #  and save MNIST as mnist.npy
    @staticmethod
    def decompress_convert_save(
            download_path=os.path.join(os.path.dirname(__file__), 'temp'),
            save_path=os.path.dirname(__file__)):
        """
        download_path (str): path of downloaded raw MNIST dataset in IDX
        format
        save_path (str): path at which processed npy file will be saved
        """
        # Gunzip, parse, and save as .npy
        # Format of .npy:
        # After loading data = np.load(), data is a np.array of np.arrays.
        # train_imgs = data[0][0]; shape = 60000 x 28 x 28
        # test_imgs = data[1][0]; shape = 10000 x 28 x 28
        # train_labels = data[0][1]; shape = 60000 x 1
        # test_labels = data[1][1]; shape = 10000 x 1
        # save as 'mnist.npy' in save_path
        pass

    @property
    def train_images(self):
        return self.data[0][0].reshape((60000, 784))

    @property
    def test_images(self):
        return self.data[1][0].reshape((10000, 784))

    @property
    def images(self):
        return np.vstack((self.train_images, self.test_images))

    @property
    def train_labels(self):
        return self.data[0][1]

    @property
    def test_labels(self):
        return self.data[1][1]

    @property
    def labels(self):
        return np.hstack((self.train_labels, self.test_labels))
