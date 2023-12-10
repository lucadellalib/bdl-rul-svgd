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

"""Dense3 (D3) model."""

# To test the implementation, open a terminal and run:
# python dense3.py

from typing import Tuple

import torch
from torch import Tensor, nn


__all__ = [
    "Dense3",
]


class Dense3(nn.Module):
    """Dense3 (D3) model, consisting of three dense
    hidden layers with sigmoid activation function.

    References
    ----------
    .. [1] M. Benker, L. Furtner, T. Semm, and M. F. Zaeh.
           "Utilizing uncertainty information in remaining useful life estimation
           via Bayesian neural networks and Hamiltonian Monte Carlo".
           In: Journal of Manufacturing Systems. 2021, pp. 799-807.
           URL: https://doi.org/10.1016/j.jmsy.2020.11.005

    Examples
    --------
    >>> import torch
    >>>
    >>>
    >>> batch_size = 10
    >>> in_shape = (2, 30, 40)
    >>> model = Dense3(in_shape)
    >>> input = torch.rand(batch_size, *in_shape)
    >>> output = model(input)

    """

    # override
    def __init__(self, in_shape: "Tuple[int, ...]") -> "None":
        """Initialize the object.

        Parameters
        ----------
        in_shape:
            The input event shape.

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        """
        if len(in_shape) < 1:
            raise ValueError(
                f"Length of `in_shape` ({in_shape}) must be in the integer interval [1, inf)"
            )

        super().__init__()
        self.in_shape = in_shape = torch.Size(in_shape)
        self.layers = nn.Sequential(
            nn.Flatten(start_dim=-len(in_shape)),
            nn.Linear(in_features=in_shape.numel(), out_features=100),
            nn.Sigmoid(),
            nn.Linear(in_features=100, out_features=100),
            nn.Sigmoid(),
            nn.Linear(in_features=100, out_features=100),
            nn.Sigmoid(),
            nn.Linear(in_features=100, out_features=1),
        )

    # override
    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        In the following, let `B` the batch size, and
        `I = {I_1, ..., I_n}` the input event shape.

        Parameters
        ----------
        input:
            The input, shape: ``[B, *I]``.

        Returns
        -------
            The output, shape: ``[B, 1]``.

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        """
        in_shape = input.shape[-len(self.in_shape) :]
        if in_shape != self.in_shape:
            raise RuntimeError(
                f"Event shape of `input` ({in_shape}) must be equal to the "
                f"given `in_shape` initialization argument ({self.in_shape})"
            )
        output = self.layers(input)
        return output


if __name__ == "__main__":
    batch_size = 10
    in_shape = (2, 30, 40)
    model = Dense3(in_shape)
    input = torch.rand(batch_size, *in_shape)
    output = model(input)
    print(model)
    print(f"Batch size: {batch_size}")
    print(f"Input shape: {in_shape}")
    print(f"Output shape: {output.shape}")
