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

"""Utilities."""

# To test the implementation, open a terminal and run:
# python utils.py

import os
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, rc
from matplotlib.legend_handler import HandlerStem
from numpy import ndarray
from scipy import stats


__all__ = [
    "plot_distributions",
    "plot_heatmap",
    "plot_metrics_comparison",
]


matplotlib.use("pdf")


def plot_metrics_comparison(
    output_image: "str",
    num_samples: "int" = 10,
    title: "str" = "",
    figsize: "Tuple[float, float]" = (6.0, 5.0),
    usetex: "bool" = False,
    style_file_or_name: "str" = "classic",
) -> "None":
    """Plot metrics (RMSE, MAE, and score) comparison.

    Parameters
    ----------
    output_image:
        The path to the output image.
    num_samples:
        The number of samples over which the
        error is averaged.
    title:
        The plot title.
    figsize:
        The figure size.
    usetex:
        True to render text with LaTeX, False otherwise.
    style_file_or_name:
        The path to a Matplotlib style file or the
        name of one of Matplotlib built-in styles
        (see https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).

    Examples
    --------
    >>> plot_metrics_comparison("metrics_comparison.png", num_samples=10)

    """
    if os.path.isfile(style_file_or_name):
        style_file_or_name = os.path.realpath(style_file_or_name)
    errors = np.linspace(-1.0, 1.0, num=10000)
    deltas = num_samples * errors
    rmses = np.sqrt(deltas**2 / num_samples)
    maes = np.abs(deltas) / num_samples
    scores = np.exp(np.where(deltas < 0.0, -deltas / 13.0, deltas / 10.0)) - 1
    with plt.style.context(style_file_or_name):
        rc("text", usetex=usetex)
        rc("font", family="serif", serif=["Computer Modern"], size=14)
        rc("axes", labelsize=16)
        rc("legend", fontsize=12.5, handletextpad=0.3, handlelength=1)
        fig, ax = plt.subplots(figsize=figsize)
        plt.plot(errors, maes, label="RMSE")
        plt.plot(errors, rmses, label="MAE")
        plt.plot(errors, scores, label="Score")
        plt.tick_params(direction="out", top=False, right=False)
        plt.title(title)
        plt.xlabel(f"Error averaged over {num_samples} samples")
        plt.ylabel("Metric")
        plt.xlim(-1.0, 1.0)
        plt.ylim(0.0, 4.0)
        plt.legend(loc="upper left", fancybox=True)
        plt.grid()
        fig.tight_layout()
        plt.savefig(output_image, bbox_inches="tight")
        plt.close()


def plot_heatmap(
    data: "ndarray",
    output_image: "str",
    title: "str" = "",
    figsize: "Tuple[float, float]" = (6.0, 5.0),
    usetex: "bool" = False,
    style_file_or_name: "str" = "classic",
    heatmap_kwargs: "Optional[Dict[str, Any]]" = None,
    xticks_kwargs: "Optional[Dict[str, Any]]" = None,
    yticks_kwargs: "Optional[Dict[str, Any]]" = None,
) -> "None":
    """Plot rectangular data as a heatmap.

    In the following, let `N` denote the number of
    rows in `data` and `M` the number of columns
    in data.

    Parameters
    ----------
    data:
        The rectangular data to plot,
        shape: ``[N, M]``.
    output_image:
        The path to the output image.
    title:
        The plot title.
    figsize:
        The figure size.
    usetex:
        True to render text with LaTeX, False otherwise.
    style_file_or_name:
        The path to a Matplotlib style file or the
        name of one of Matplotlib built-in styles
        (see https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).
    heatmap_kwargs:
        The keyword arguments to pass to `seaborn.heatmap`.
        Default to ``{}``.
    xticks_kwargs:
        The keyword arguments to pass to `matplotlib.pyplot.xticks`.
        Default to ``{}``.
    yticks_kwargs:
        The keyword arguments to pass to `matplotlib.pyplot.yticks`.
        Default to ``{}``.

    Examples
    --------
    >>> import numpy as np
    >>>
    >>>
    >>> data = np.random.rand(30, 14)
    >>> plot_heatmap(data, "data.png")

    """
    if os.path.isfile(style_file_or_name):
        style_file_or_name = os.path.realpath(style_file_or_name)
    heatmap_kwargs = heatmap_kwargs or {}
    xticks_kwargs = xticks_kwargs or {}
    yticks_kwargs = yticks_kwargs or {}
    with plt.style.context(style_file_or_name):
        rc("text", usetex=usetex)
        rc("font", family="serif", serif=["Computer Modern"], size=14)
        rc("axes", labelsize=16)
        rc("legend", fontsize=12.5, handletextpad=0.3, handlelength=1)
        fig, ax = plt.subplots(figsize=figsize)
        heatmap_kwargs.setdefault("vmin", -1.0)
        heatmap_kwargs.setdefault("vmax", 1.0)
        heatmap_kwargs.setdefault("center", 0.0)
        heatmap_kwargs.setdefault("robust", True)
        heatmap_kwargs.setdefault("cbar_kws", {"shrink": 0.8})
        heatmap_kwargs.setdefault("square", True)
        xticks_kwargs.setdefault("ticks", [0.5, 4.5, 9.5, 13.5])
        xticks_kwargs.setdefault("labels", [1, 5, 10, 14])
        xticks_kwargs.setdefault("rotation", 0)
        yticks_kwargs.setdefault("ticks", [0.5, 6.5, 12.5, 17.5, 23.5, 29.5])
        yticks_kwargs.setdefault("labels", [1, 7, 13, 18, 24, 30])
        yticks_kwargs.setdefault("rotation", 0)
        sns.heatmap(data, **heatmap_kwargs)
        plt.xticks(**xticks_kwargs)
        plt.yticks(**yticks_kwargs)
        plt.tick_params(direction="out", top=False, right=False)
        plt.title(title)
        plt.xlabel("Feature")
        plt.ylabel("Time step")
        fig.axes[-1].tick_params(
            direction="in", top=False, right=False, left=False, bottom=False
        )
        fig.tight_layout()
        plt.savefig(output_image, bbox_inches="tight")
        plt.close()
        del data, fig, ax


def plot_distributions(
    samples: "Dict[str, ndarray]",
    output_image: "str",
    distribution_types: "Optional[Dict[str, str]]" = None,
    vlines: "Optional[Sequence[Any]]" = None,
    xlabel: "Optional[str]" = None,
    ylabel: "Optional[str]" = None,
    xlims: "Optional[Tuple[float, float]]" = None,
    ylims: "Optional[Tuple[float, float]]" = None,
    title: "str" = "",
    figsize: "Tuple[float, float]" = (6.0, 5.0),
    usetex: "bool" = False,
    style_file_or_name: "str" = "classic",
) -> "None":
    """Plot distributions via kernel density
    estimation based on samples drawn from them.

    In the following, let `N` denote the number
    of samples.

    Parameters
    ----------
    samples:
        The samples, i.e. a dict that maps distribution
        names to their corresponding samples. Shape of
        ``samples[name]``: ``[N, 1]`` or ``[N, 2]``.
    output_image:
        The path to the output image.
    distribution_types:
        The distribution types, i.e. a dict that maps
        distribution names to their corresponding types.
    vlines:
        The sequence of keyword arguments to
        pass to `matplotlib.pyplot.axvline`
        for plotting extra vertical lines.
        Default to ``[]``.
    xlabel:
        The x-axis label.
        Default to "Value" if shape of sample
        is equal to ``[N, 1]``, "x" otherwise.
    ylabel:
        The y-axis label.
        Default to "Probability density" if shape of
        sample is equal to ``[N, 1]``, "y" otherwise.
    xlims:
        The x-axis limits.
    ylims:
        The y-axis limits.
    title:
        The plot title.
    figsize:
        The figure size.
    usetex:
        True to render text with LaTeX, False otherwise.
    style_file_or_name:
        The path to a Matplotlib style file or the
        name of one of Matplotlib built-in styles
        (see https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).

    Raises
    ------
    ValueError
        If an invalid argument value is given.

    Examples
    --------
    >>> import numpy as np
    >>>
    >>>
    >>> samples = np.random.normal(0, 1, size=(100, 1))
    >>> plot_distributions({"": samples}, "distribution.png")

    """
    if os.path.isfile(style_file_or_name):
        style_file_or_name = os.path.realpath(style_file_or_name)
    if distribution_types is None:
        distribution_types = {}
    if vlines is None:
        vlines = []
    with plt.style.context(style_file_or_name):
        rc("text", usetex=usetex)
        rc("font", family="serif", serif=["Computer Modern"], size=14)
        rc("axes", labelsize=16)
        rc("legend", fontsize=12.5, handletextpad=0.3, handlelength=1)
        fig, ax = plt.subplots(figsize=figsize)
        legend_kwargs = {}
        legend_kwargs["handler_map"] = {}
        for k, v in samples.items():
            if v.ndim != 2 or v.shape[1] > 2:
                raise ValueError(
                    f"`samples[{k}].ndim` ({v.ndim}) must be equal to 2 and "
                    f"`samples[{k}].shape[1]` ({v.shape[1]}) must be equal to 1 or 2"
                )
            label = k or None
            if v.shape[1] == 1:
                if len(v) > 1:
                    distribution_type = distribution_types.get(k, None)
                    if distribution_type == "normal":
                        mean = v[:, 0].mean()
                        stddev = v[:, 0].std()
                        support = np.linspace(
                            mean - 4 * stddev, mean + 4 * stddev, 1000
                        )
                        plt.plot(
                            support, stats.norm.pdf(support, mean, stddev), label=label
                        )
                        del support
                    else:
                        sns.kdeplot(v[:, 0], label=label)
                else:
                    stem = plt.stem(v, 1.0, markerfmt="^", basefmt=" ", label=label)
                    plt.ylim(0.0, 1.1 * ax.get_ylim()[-1])
                    legend_kwargs["handler_map"].update(
                        {stem: HandlerStem(numpoints=1)}
                    )
                    del stem
                if xlabel is None:
                    xlabel = "Value"
                if ylabel is None:
                    ylabel = "Probability density"
            elif v.shape[1] == 2:
                if len(v) > 1:
                    sns.kdeplot(x=v[:, 0], y=v[:, 1], fill=True, label=label)
                else:
                    plt.scatter(v[:, 0], v[:, 1], label=label)
                if xlabel is None:
                    xlabel = "x"
                if ylabel is None:
                    ylabel = "y"
        for vline in vlines:
            plt.axvline(**vline)
        plt.tick_params(direction="out", top=False, right=False)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if xlims:
            plt.xlim(*xlims)
        if ylims:
            plt.ylim(*ylims)
        handles, labels = ax.get_legend_handles_labels()
        if any(handles):
            sorted_handles, sorted_labels = [], []
            for k in samples:
                label = k or None
                sorted_labels.append(label)
                sorted_handles.append(handles[labels.index(label)])
            for k, v in zip(labels, handles):
                if k not in sorted_labels:
                    sorted_labels.append(k)
                    sorted_handles.append(v)
            plt.legend(
                sorted_handles,
                sorted_labels,
                loc="upper left",
                fancybox=True,
                **legend_kwargs,
            )
            del sorted_handles, sorted_labels
        plt.grid()
        fig.tight_layout()
        plt.savefig(output_image, bbox_inches="tight")
        plt.close()
        del samples, vlines, fig, ax, handles, labels


if __name__ == "__main__":
    output_dir = "images"
    os.makedirs(output_dir, exist_ok=True)
    plot_metrics_comparison(
        os.path.join(output_dir, "metrics_comparison.pdf"), usetex=True
    )
