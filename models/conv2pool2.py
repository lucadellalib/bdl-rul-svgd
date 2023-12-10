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

"""Conv2Pool2 (C2P2) model."""

# To test the implementation, open a terminal and run:
# python conv2pool2.py

from typing import Tuple

import torch
from torch import Tensor, nn


__all__ = [
    "Conv2Pool2",
]


class Conv2Pool2(nn.Module):
    """Conv2Pool2 (C2P2) model, consisting of two 2D convolutional
    hidden layers with sigmoid activation function interspersed
    with 2D average pooling hidden layers.

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
    >>> model = Conv2Pool2(in_shape)
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
            nn.Conv2d(in_channels=in_shape[0], out_channels=8, kernel_size=(5, 14)),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(2, 1)),
            nn.Conv2d(in_channels=8, out_channels=14, kernel_size=(2, 1)),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(2, 1)),
            nn.Flatten(start_dim=-len(in_shape)),
        )
        example_input = torch.rand(in_shape)
        example_output = self.layers(example_input)
        self.layers.append(
            nn.Linear(in_features=example_output.numel(), out_features=1)
        )

    # override
    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        In the following, let `B` the batch size, and
        `I = {C, H, W}` the input event shape.

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
    model = Conv2Pool2(in_shape)
    input = torch.rand(batch_size, *in_shape)
    output = model(input)
    print(model)
    print(f"Batch size: {batch_size}")
    print(f"Input shape: {in_shape}")
    print(f"Output shape: {output.shape}")
