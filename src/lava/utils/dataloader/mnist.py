# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import os
import numpy as np


class MnistDataset:
    mirrors = [
        "http://yann.lecun.com/exdb/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
    ]

    files = [
        "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",
    ]

    def __init__(self, data_path=os.path.join(os.path.dirname(__file__),
                                              'mnist.npy')):
        """data_path (str): Path to mnist.npy file containing the MNIST
        dataset"""
        if not os.path.exists(data_path):
            # Download MNIST from internet and convert it to .npy
            os.makedirs(os.path.join(os.path.dirname(__file__), 'temp'),
                        exist_ok=True)
            MnistDataset. \
                download_mnist(path=os.path.join(
                    os.path.dirname(__file__),
                    'temp'
                )
                )
            # GUnzip, Parse and save MNIST data as .npy
            MnistDataset.decompress_convert_save(
                download_path=os.path.join(os.path.dirname(__file__), 'temp'),
                save_path=data_path)
        self.data = np.load(data_path, allow_pickle=True)

    @staticmethod
    def download_mnist(path=os.path.join(os.path.dirname(__file__), 'temp')):
        import urllib.request
        import urllib.error

        for file in MnistDataset.files:
            err = None
            for mirror in MnistDataset.mirrors:
                try:
                    url = f"{mirror}{file}"
                    if url.lower().startswith("http"):
                        # Disabling security lint because we are using hardcoded
                        # URLs specified above
                        res = urllib.request.urlopen(url)  # nosec # noqa
                        with open(os.path.join(path, file), "wb") as f:
                            f.write(res.read())
                        break
                    else:
                        raise ValueError(f"Specified URL ({url}) does not "
                                         "start with 'http'.")
                except urllib.error.URLError as exception:
                    err = exception
                    continue
            else:
                print("Failed to download mnist dataset")
                raise err

    @staticmethod
    def decompress_convert_save(
            download_path=os.path.join(os.path.dirname(__file__), 'temp'),
            save_path=os.path.dirname(__file__)):
        """
        download_path (str): path of downloaded raw MNIST dataset in IDX
        format
        save_path (str): path at which processed npy file will be saved

        After loading data = np.load(), data is a np.array of np.arrays.
        train_imgs = data[0][0]; shape = 60000 x 28 x 28
        test_imgs = data[1][0]; shape = 10000 x 28 x 28
        train_labels = data[0][1]; shape = 60000 x 1
        test_labels = data[1][1]; shape = 10000 x 1
        """

        import gzip
        arrays = []
        for file in MnistDataset.files:
            with gzip.open(os.path.join(download_path, file), "rb") as f:
                if "images" in file:
                    arr = np.frombuffer(f.read(), np.uint8, offset=16)
                    arr = arr.reshape(-1, 28, 28)
                else:
                    arr = np.frombuffer(f.read(), np.uint8, offset=8)
                arrays.append(arr)

        np.save(
            save_path,
            np.array(
                [[arrays[0], arrays[1]], [arrays[2], arrays[3]]],
                dtype="object"),
        )

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
