# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty


def sub_sample(data: np.ndarray,
               indices: np.ndarray,
               max_events: int,
               random_rng: ty.Optional[np.random.Generator] = None) \
            -> ty.Tuple[np.ndarray, np.ndarray]:
    data_idx_array = np.arange(0, data.shape[0])
    sampled_idx = random_rng.choice(data_idx_array,
                                    max_events,
                                    replace=False)

    sampled_idx = np.sort(sampled_idx)
    return data[sampled_idx], indices[sampled_idx]


def encode_data_and_indices(frame_shape: ty.Tuple,
                             events: np.ndarray) \
           -> ty.Tuple[np.ndarray, np.ndarray]:
    """Extracts the polarity data, and x and y indices from the given
    batch of events, and encodes them using C-style encoding."""
    xs, ys, ps = events['x'], events['y'], events['polarity']
    data = ps
    indices = np.ravel_multi_index((xs, ys), frame_shape)

    return data, indices
