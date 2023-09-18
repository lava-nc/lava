# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import glob
from PIL import Image
from sklearn.model_selection import train_test_split

try:
    import torch
    from torch.utils.data import DataLoader

    import torchvision.transforms as transforms
    import torchvision.models as models

    LIBS_ARE_AVAILABLE = True

except ModuleNotFoundError:
    LIBS_ARE_AVAILABLE = False


class Coil100Dataset:
    def __init__(self, root_dir, obj_list=np.arange(100), transform=None,
                 size=64, train=True, test_size=None, seed=1234):
        """
        this function builds a data frame which contains the path to image
        and the tag/object name using the prefix of the image name
        """
        self.targets = []
        self.paths = []
        self.size = size
        self.root_dir = root_dir
        self.transform = transform

        path = root_dir + '/coil-100/*.png'

        # list files
        files = glob.glob(path)

        for file in files:
            self.targets.append(
                int(file.split("/")[-1].split("__")[0].split("j")[1]) - 1)
            self.paths.append(file)

        self.targets = np.array(self.targets)
        self.paths = np.array(self.paths)

        sorted_inds = np.argsort(self.targets)
        self.targets = self.targets[sorted_inds]
        self.paths = self.paths[sorted_inds]

        obj_slices = np.concatenate(
            [np.arange(ind * 72, (ind + 1) * 72) for ind in obj_list])
        n_classes = len(obj_list)
        # self.targets = self.targets[obj_slices]
        self.targets = self.targets[:72 * n_classes]
        self.paths = self.paths[obj_slices]

        if test_size is not None:
            train_indices, test_indices, _, _ = train_test_split(
                range(len(self.targets)),
                self.targets,
                stratify=self.targets,
                test_size=test_size,
                random_state=seed,
                shuffle=True)
            if train:
                self.targets = self.targets[train_indices]
                self.paths = self.paths[train_indices]
            else:
                self.targets = self.targets[test_indices]
                self.paths = self.paths[test_indices]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):

        img_path, img_label = self.paths[idx], self.targets[idx]
        img = Image.open(img_path)
        if self.transform:
            img = img.resize(size=(self.size, self.size))
            img = self.transform(img)
        return img, img_label


def extract_and_load_features(extract=False):
    print("LIBS_ARE_AVAILABLE: ", LIBS_ARE_AVAILABLE)
    if LIBS_ARE_AVAILABLE and extract:
        dataset_dir = 'datasets/'

        device = "cuda" if torch.cuda.is_available() else "cpu"

        test_size = None

        coil100 = Coil100Dataset(root_dir=dataset_dir,
                                 transform=transforms.ToTensor(), size=64,
                                 train=False, test_size=test_size)

        feat_ext_dl = DataLoader(coil100, batch_size=1, shuffle=False,
                                 num_workers=8)

        feat_ext = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        feat_ext = torch.nn.Sequential(*list(feat_ext.children())[:-1])
        feat_ext = feat_ext.to(device)
        feat_ext.eval()

        embeddings = []
        with torch.no_grad():
            for batch in feat_ext_dl:
                image, label = batch
                image, label = image.to(device), label.to(device)
                emb = feat_ext(image).flatten(start_dim=1)
                embeddings.append(emb)

        embeddings = torch.cat(embeddings, 0)
        X = np.array(torch.Tensor.cpu(embeddings))
        y = coil100.targets

    else:
        with open('datasets/coil_100_features/coil_100_features_effnet.npy',
                  'rb') as f:
            X = np.load(f)
            y = np.load(f)

    return X, y


def wta_hyperparam_search(a_in,
                          n_steps,
                          du_values=np.arange(10, 800, 20),
                          dv_values=np.arange(40, 800, 20),
                          v_th_values=np.arange(50000, 80000, 1000)):
    # Input values to differentiate based on spike time
    n_steps = n_steps
    # Activation inputs to be distinguished in the temporal encoding by the
    # spike times
    a_in = a_in
    # Hyperparameter grid to search
    du_values = du_values
    dv_values = dv_values
    v_th_values = v_th_values

    num_neurons = len(a_in)

    # Perform grid search
    best_params = np.zeros(shape=(3,), dtype=np.int32)
    best_spike_times = 50 * np.ones(shape=(num_neurons,), dtype=np.int32)

    for du, dv, vth in product(du_values, dv_values, v_th_values):

        v_hist = loihi_lif_bit_acc_sim(du, dv, a_in, n_steps)
        crossing_times = np.argmax(v_hist > vth, axis=0)
        unique_check = len(set(crossing_times)) == len(crossing_times)
        # Check if current parameter combination is the best so far
        if unique_check and crossing_times[0] == 0:

            if np.max(crossing_times) < np.max(best_spike_times):
                print("-----------------------------")
                print("du =", du)
                print("dv =", dv)
                print("vth =", vth)
                best_spike_times = crossing_times.copy()

                print("The Latest spike time:", np.max(crossing_times))
                best_params = (du, dv, vth)

    # Print spike times and best parameter combination
    for i in range(num_neurons):
        print(
            f"Spike time for neuron {i + 1} with a_in={a_in[i]}:"
            f" {best_spike_times[i]}")

    print("Best Parameter Combination:")
    print("du =", best_params[0])
    print("dv =", best_params[1])
    print("vth =", best_params[2])

    return best_params


def loihi_lif_bit_acc_sim(du, dv, a_in, n_steps):
    # Loihi neuron hardware params
    ds_offset = 1
    dm_offset = 0
    effective_bias = 0
    # Let's define some bit-widths from Loihi
    # State variables u and v are 24-bits wide
    uv_bitwidth = 24
    max_uv_val = 2 ** (uv_bitwidth - 1)
    # Decays need an MSB alignment with 12-bits
    decay_shift = 12
    decay_unity = 2 ** decay_shift
    # Threshold and incoming activation are MSB-aligned using 6-bits
    act_shift = 6

    num_neurons = len(a_in)
    a_in_list = np.zeros(shape=(n_steps, num_neurons), dtype=np.int32)
    a_in_list[0, :] = a_in

    # Reset neuron variables for each parameter combination
    v_hist = np.zeros(shape=(n_steps, num_neurons), dtype=np.int32)
    v = np.zeros(shape=(1, num_neurons), dtype=np.int32)
    u = np.zeros(shape=(1, num_neurons), dtype=np.int32)

    for t in range(n_steps):
        decay_const_u = du + ds_offset
        decayed_curr = np.int64(u) * (decay_unity - decay_const_u)
        decayed_curr = np.sign(decayed_curr) * np.right_shift(
            np.abs(decayed_curr), decay_shift
        )
        decayed_curr = np.int32(decayed_curr)
        activation_in = np.left_shift(a_in_list[t, :], act_shift)
        decayed_curr += activation_in
        wrapped_curr = np.where(
            decayed_curr > max_uv_val,
            decayed_curr - 2 * max_uv_val,
            decayed_curr,
        )
        wrapped_curr = np.where(
            wrapped_curr <= -max_uv_val,
            decayed_curr + 2 * max_uv_val,
            wrapped_curr,
        )
        u[:] = wrapped_curr
        decay_const_v = dv + dm_offset

        neg_voltage_limit = -np.int32(max_uv_val) + 1
        pos_voltage_limit = np.int32(max_uv_val) - 1

        decayed_volt = np.int64(v) * (decay_unity - decay_const_v)
        decayed_volt = np.sign(decayed_volt) * np.right_shift(
            np.abs(decayed_volt), decay_shift
        )
        decayed_volt = np.int32(decayed_volt)
        updated_volt = decayed_volt + u + effective_bias
        v[:] = np.clip(updated_volt, neg_voltage_limit, pos_voltage_limit)

        v_hist[t, :] = (v / 64)

    return v_hist


def plot_wta_voltage_dynamics(du, dv, vth, a_in, n_steps):
    v_hist = loihi_lif_bit_acc_sim(du, dv, a_in, n_steps)

    spike_times = np.argmax(v_hist > vth, axis=0)
    n_steps = v_hist.shape[0]
    n_input = len(spike_times)
    v_min, v_max = np.min(v_hist), np.max(v_hist)

    plt.figure(figsize=(6, 6))
    plt.step(np.arange(n_steps + 1),
             np.vstack((np.zeros((1, n_input)), v_hist)))
    ax = plt.gca()
    ax.hlines(y=vth, xmin=0, xmax=n_steps, linewidth=3, color='grey',
              linestyle='--')
    for t in spike_times:
        ax.vlines(x=t, ymin=v_min, ymax=vth, linewidth=2, color='lightgrey',
                  linestyle='--')
    plt.xticks(np.arange(n_steps), size=10)
    plt.yticks(np.arange(np.round(v_min, -3), np.round(1.05 * v_max, -3),
                         np.round((1.05 * v_max - v_min) / 10, -3)), size=10)
    plt.ylim([v_min, 1.05 * v_max])
    plt.xlim([0, n_steps])
    plt.xlabel("Time", size=15)
    plt.ylabel("Voltage", size=15)
    legend = []
    for i in range(v_hist.shape[1]):
        legend.append(f"a_in={v_hist[0, i]}")
    legend.append("v_th")
    plt.legend(legend, loc='lower right', fontsize=8)
    plt.title("Voltage dynamics of LIF neuron tuned for temporal WTA",
              fontsize=15)
    plt.show()


def plot_spikes(spikes, figsize, legend, colors, title, num_steps):
    offsets = list(range(1, len(spikes) + 1))
    num_x_ticks = np.arange(0, num_steps + 1, 25)

    plt.figure(figsize=figsize)

    plt.eventplot(positions=spikes,
                  lineoffsets=offsets,
                  linelength=0.9,
                  colors=colors)

    plt.title(title)
    plt.xlabel("Time steps")
    plt.ylabel("Neurons")

    plt.xticks(num_x_ticks)
    plt.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='lightgray', linewidth=0.8)
    plt.minorticks_on()

    plt.yticks(ticks=offsets, labels=legend)

    plt.show()


def plot_time_series(time, time_series, ylabel, title, figsize, color):
    plt.figure(figsize=figsize)
    plt.step(time, time_series, color=color)

    plt.title(title)
    plt.xlabel("Time steps")
    plt.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='lightgray', linewidth=0.8)
    plt.minorticks_on()

    plt.ylabel(ylabel)

    plt.show()


def plot_multiple_time_series(time, time_series_list, ylabel,
                              title, figsize, color, legend,
                              leg_loc="upper left"):
    plt.figure(figsize=figsize)
    for i in range(time_series_list.shape[1]):
        plt.step(time, time_series_list[:, i], label=legend[i], color=color[i])

    plt.title(title)
    plt.xlabel("Time steps")
    plt.ylabel(ylabel)
    plt.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='lightgray', linewidth=0.8)
    plt.minorticks_on()
    plt.xlim(0, time_series_list.shape[0])

    plt.legend(loc=leg_loc)

    plt.show()


def plot_spikes_time_series(time, time_series, spikes, figsize, legend,
                            colors, y_label_time_series, title, num_steps):
    offsets = list(range(1, len(spikes) + 1))
    num_x_ticks = np.arange(0, num_steps + 1, 25)

    plt.figure(figsize=figsize)

    plt.subplot(211)
    plt.eventplot(positions=spikes,
                  lineoffsets=offsets,
                  linelength=0.9,
                  colors=colors[:-1])

    plt.title("Spike Arrival")
    plt.xlabel("Time steps")

    plt.xticks(num_x_ticks)
    plt.xlim(0, num_steps)
    plt.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='lightgray', linewidth=0.8)
    plt.minorticks_on()

    plt.yticks(ticks=offsets, labels=legend)
    plt.tight_layout(pad=3.0)

    plt.subplot(212)
    plt.step(time, time_series)

    plt.title(title[0])
    plt.xlabel("Time steps")
    plt.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='lightgray', linewidth=0.8)
    plt.minorticks_on()
    plt.margins(x=0)

    plt.ylabel(y_label_time_series)

    plt.show()
