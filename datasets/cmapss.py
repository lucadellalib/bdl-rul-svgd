#!/usr/bin/env python3

# ==============================================================================
# Copyright 2022 Luca Della Libera and others.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset."""

# To test the implementation, open a terminal and run:
# python cmapss.py

import os
import tarfile
from typing import Tuple

import numpy as np
from numpy import ndarray
from torch.utils.data import Dataset


__all__ = [
    "CMAPSS",
]


class CMAPSS(Dataset):
    """Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset.

    References
    ----------
    .. [1] A. Saxena, K. Goebel, D. Simon, and N. Eklund.
           "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation".
           In: International Conference on Prognostics and Health Management. 2008, pp. 1-9.
           URL: https://ieeexplore.ieee.org/abstract/document/4711414

    Examples
    --------
    >>> import os
    >>>
    >>>
    >>> current_dir = os.path.realpath(os.path.dirname(__file__))
    >>> root_dir = os.path.dirname(current_dir)
    >>> data_archive = os.path.join(root_dir, "data", "cmapss.tar.gz")
    >>> dataset = CMAPSS(
    ...     data_archive,
    ...     subset="fd001",
    ...     split="train",
    ...     window_size=30,
    ...     normalization="min-max",
    ... )
    >>> num_chunks = len(dataset)
    >>> first_chunk, first_rul = dataset[0]
    >>> chunk_shape = first_chunk.shape
    >>> rul_shape = first_rul.shape
    >>> for chunk, rul in dataset:
    ...     pass

    """

    def __init__(
        self,
        data_archive: "str",
        subset: "str" = "fd001",
        split: "str" = "train",
        window_size: "int" = 30,
        normalization: "str" = "min-max",
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        data_archive:
            The path to the TAR.GZ archive containing the compressed data.
        subset:
            The data subset ("fd001", "fd002", "fd003" or "fd004").
        split:
            The data split ("train" or "test").
        window_size:
            The window size for sliding window segmentation.
        normalization:
            The normalization strategy ("min-max" or "z-score").

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        """
        subset = subset.lower()
        if subset not in ["fd001", "fd002", "fd003", "fd004"]:
            raise ValueError(
                f"`subset` ({subset}) must be one of {['fd001', 'fd002', 'fd003', 'fd004']}"
            )
        if split not in ["train", "test"]:
            raise ValueError(f"`split` ({split}) must be one of {['train', 'test']}")
        if window_size < 1 or not float(window_size).is_integer():
            raise ValueError(
                f"`window_size` ({window_size}) must be in the integer interval [1, inf)"
            )
        if normalization not in ["min-max", "z-score"]:
            raise ValueError(
                f"`normalization` ({normalization}) must be one of {['min-max', 'z-score']}"
            )

        self.data_archive = data_archive
        self.subset = subset
        self.split = split
        self.window_size = window_size = int(window_size)
        self.normalization = normalization
        self._data = []  # Store data (privately)

        # Extract compressed data
        with tarfile.open(data_archive) as tar_archive:
            # Each row is a snapshot of data taken during a single operational cycle,
            # each column is a different variable. The columns correspond to:
            # 1)  unit number (or trajectory number, since only one trajectory is collected for each unit)
            # 2)  time, in cycles
            # 3)  operational setting 1
            # 4)  operational setting 2
            # 5)  operational setting 3
            # 6)  sensor measurement  1
            # 7)  sensor measurement  2
            # ...
            # 26) sensor measurement  26
            trajectories_file = tar_archive.extractfile(f"{split}_{subset.upper()}.txt")

            # Always load the corresponding train file to compute normalization statistics
            # NOTE: the same normalizer fitted on train data must be applied to test data
            train_trajectories_file = tar_archive.extractfile(
                f"train_{subset.upper()}.txt"
            )

            # Each row is a single integer representing the remaining useful life
            # of the corresponding test trajectory (used only for "test" split)
            test_ruls_file = tar_archive.extractfile(f"RUL_{subset.upper()}.txt")

            # Read files into NumPy arrays
            trajectories = np.loadtxt(trajectories_file)
            train_trajectories = np.loadtxt(train_trajectories_file)
            test_ruls = np.loadtxt(test_ruls_file)

        # Compute length of each trajectory
        trajectory_lengths = np.unique(trajectories[:, 0], return_counts=True)[1]
        if (trajectory_lengths < window_size).any():
            raise ValueError(
                f"`window_size` {(window_size)} must be greater than or equal "
                f"to the minimum trajectory length ({trajectory_lengths.min()})"
            )

        # Remove trajectory number and time columns
        trajectories = np.delete(trajectories, [0, 1], axis=1)
        train_trajectories = np.delete(train_trajectories, [0, 1], axis=1)

        if subset in ["fd001", "fd003"]:
            # Remove operational setting columns (useless for prediction)
            trajectories = np.delete(trajectories, [0, 1, 2], axis=1)
            train_trajectories = np.delete(train_trajectories, [0, 1, 2], axis=1)

            # Remove constant sensor measurement columns (useless for prediction)
            trajectories = np.delete(trajectories, [0, 4, 5, 9, 15, 17, 18], axis=1)
            train_trajectories = np.delete(
                train_trajectories, [0, 4, 5, 9, 15, 17, 18], axis=1
            )

        # Compute normalization statistics on train
        # data and define corresponding normalizer
        if normalization == "min-max":
            self.stats = stats = {
                "min": train_trajectories.min(axis=0),
                "max": train_trajectories.max(axis=0),
            }
            self.normalizer = lambda x: -1 + 2 * (x - stats["min"]) / (
                stats["max"] - stats["min"]
            )
        elif normalization == "z-score":
            self.stats = stats = {
                "mean": train_trajectories.mean(axis=0),
                "std": train_trajectories.std(axis=0),
            }
            self.normalizer = lambda x: (x - stats["mean"]) / stats["std"]

        # Apply normalization to current data (might be test
        # data depending on the initialization arguments)
        trajectories = self.normalizer(trajectories)

        # Iterate through all trajectories
        start_idx = 0
        for trajectory_id, trajectory_length in enumerate(trajectory_lengths):
            end_idx = start_idx + trajectory_length
            trajectory = trajectories[start_idx:end_idx, :]

            # For each trajectory, extract chunks via sliding window
            # segmentation (starting from the end of the trajectory)
            num_chunks = trajectory_length - window_size + 1
            for rul in range(num_chunks):
                j = trajectory_length - rul
                chunk = trajectory[j - window_size : j]

                # Add channel dimension to the left
                # Shape: [C, H, W] = [1, window_size, num_features]
                chunk = chunk[None]

                # If split is "test", keep only the last chunk
                # and assign the corresponding test RUL
                if split == "test":
                    test_rul = test_ruls[trajectory_id]
                    self._data.append(
                        (
                            np.array(chunk, dtype=np.float32),
                            np.array(test_rul, dtype=np.float32),
                        )
                    )
                    break

                self._data.append(
                    (
                        np.array(chunk, dtype=np.float32),
                        np.array(rul, dtype=np.float32),
                    )
                )

            start_idx = end_idx

    # override
    def __getitem__(self, index: "int") -> "Tuple[ndarray, ndarray]":
        return self._data[index]

    def __len__(self) -> "int":
        return len(self._data)

    def __repr__(self) -> "str":
        return (
            f"{type(self).__name__}"
            f"(data_archive: {self.data_archive}, "
            f"subset: {self.subset}, "
            f"split: {self.split}, "
            f"window_size: {self.window_size}, "
            f"normalization: {self.normalization})"
        )


if __name__ == "__main__":
    current_dir = os.path.realpath(os.path.dirname(__file__))
    root_dir = os.path.dirname(current_dir)
    data_archive = os.path.join(root_dir, "data", "cmapss.tar.gz")
    print("---------------------------------------------------")
    for subset, window_size in [
        ("fd001", 30),
        ("fd002", 20),
        ("fd003", 30),
        ("fd004", 15),
    ]:
        for split in ["train", "test"]:
            dataset = CMAPSS(
                data_archive,
                subset=subset,
                split=split,
                window_size=window_size,
                normalization="min-max",
            )
            num_chunks = len(dataset)
            first_chunk, first_rul = dataset[0]
            chunk_shape = first_chunk.shape
            rul_shape = first_rul.shape
            print(dataset)
            print(f"Number of chunks: {num_chunks}")
            print(f"Chunk shape: {chunk_shape}")
            print(f"RUL shape: {rul_shape}")
            print("---------------------------------------------------")
