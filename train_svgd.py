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

"""Train a Bayesian model via Stein variational gradient descent (SVGD)."""

# To run an experiment, open a terminal and run:
# python train_svgd.py [-h] [-r RESTORE] {d3,c2p2} {fd001,fd002,fd003,fd004}

import argparse
import logging
import math
import os
from datetime import datetime
from typing import Any, Dict

import torch
from bayestorch.kernels import RBFSteinKernel
from bayestorch.losses import NLUPLoss
from bayestorch.models import ParticlePosteriorModel
from bayestorch.preconditioners import SVGD
from ray.air import Checkpoint, RunConfig, session
from ray.tune import Tuner, grid_search, with_resources
from torch import nn
from torch.utils.data import DataLoader

from datasets import CMAPSS
from models import Conv2Pool2, Dense3
from utils import plot_distributions, plot_heatmap


__all__ = [
    "run_experiment",
]


_LOGGER = logging.getLogger(__name__)

_ROOT_DIR = os.path.realpath(os.path.dirname(__file__))


def run_experiment(config: "Dict[str, Any]") -> "None":
    """Run an experiment.

    Parameters
    ----------
    config:
        The experiment configuration.

    Raises
    ------
    ValueError
        If an invalid argument value is given.

    """
    # Read experiment parameters from configuration
    seed = config["seed"]
    data_archive = config["data_archive"]
    subset = config["subset"]
    window_size = config["window_size"]
    normalization = config["normalization"]
    max_target = config["max_target"]
    test_max_targets = [
        float("inf"),
        max_target,
    ]  # Test both without and with target rectification
    model_name = config["model_name"].lower()
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    normal_prior_std = config["normal_prior_std"]
    num_particles = config["num_particles"]
    lr = config["lr"]
    step_size = config["step_size"]
    gamma = config["gamma"]
    checkpoint_freq = config["checkpoint_freq"]
    checkpoint_at_end = config["checkpoint_at_end"]
    test_only = config["test_only"]
    plot = config["plot"]

    if seed < 0 or not float(seed).is_integer():
        raise ValueError(f"`seed` ({seed}) must be in the integer interval [0, inf)")
    if model_name not in ["d3", "c2p2"]:
        raise ValueError(f"`subset` ({subset}) must be one of {['d3', 'c2p2']}")
    if num_epochs < 1 or not float(num_epochs).is_integer():
        raise ValueError(
            f"`num_epochs` ({num_epochs}) must be in the integer interval [1, inf)"
        )
    if batch_size < 1 or not float(batch_size).is_integer():
        raise ValueError(
            f"`batch_size` ({batch_size}) must be in the integer interval [1, inf)"
        )
    if normal_prior_std <= 0.0:
        raise ValueError(
            f"`normal_prior_std` ({normal_prior_std}) must be in the interval (0, inf)"
        )
    if num_particles < 1 or not float(num_particles).is_integer():
        raise ValueError(
            f"`num_particles` ({num_particles}) must be in the integer interval [1, inf)"
        )
    if lr < 0.0:
        raise ValueError(f"`lr` ({lr}) must be in the interval [0, inf)")
    if step_size < 1 or not float(step_size).is_integer():
        raise ValueError(
            f"`step_size` ({step_size}) must be in the integer interval [1, inf)"
        )
    if gamma < 0.0:
        raise ValueError(f"`gamma` ({gamma}) must be in the interval [0, inf)")
    if checkpoint_freq < 0 or not float(checkpoint_freq).is_integer():
        raise ValueError(
            f"`checkpoint_freq` ({checkpoint_freq}) must be in the integer interval [0, inf)"
        )

    seed = int(seed)
    num_epochs = int(num_epochs)
    batch_size = int(batch_size)
    num_particles = int(num_particles)
    step_size = int(step_size)
    checkpoint_freq = int(checkpoint_freq)

    # Enable reproducibility
    import torch  # Fix TypeError: cannot pickle 'CudnnModule' object

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    # See https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Set seed
    torch.manual_seed(seed)

    # Set device
    device = "cuda" if torch.cuda.is_available() > 0 else "cpu"

    # Build train dataset and dataloader
    train_dataset = CMAPSS(
        data_archive,
        subset=subset,
        split="train",
        window_size=window_size,
        normalization=normalization,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device == "cuda",
    )

    # Build model
    in_shape = train_dataset[0][0].shape
    if model_name == "d3":
        model_cls = Dense3
    elif model_name == "c2p2":
        model_cls = Conv2Pool2
    model = model_cls(in_shape)
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    model = ParticlePosteriorModel(
        model,
        torch.distributions.Normal,
        {
            "loc": torch.zeros(num_parameters),
            "scale": torch.full((num_parameters,), normal_prior_std),
        },
        num_particles=num_particles,
    ).to(device)
    print(model)
    num_parameters = sum(parameter.numel() for parameter in model.parameters())

    # Define loss function
    criterion = NLUPLoss()

    # Build optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    # Build learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma,
    )

    # Restore checkpoint if any
    start_epoch = 0
    checkpoint = session.get_checkpoint()
    if checkpoint is not None:
        checkpoint = checkpoint.to_dict()
        torch.random.set_rng_state(checkpoint["rng_state"])
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    # Performance metrics
    metrics = {}

    # Train if test_only=False
    if not test_only:
        # Build kernel
        kernel = RBFSteinKernel()

        # Build preconditioner
        preconditioner = SVGD(
            model.parameters(),
            kernel,
            num_particles,
        )

        train_loss = 0.0
        model.train()

        for epoch in range(start_epoch, num_epochs):

            # Train one epoch
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)

                # Apply target rectification
                target = target.clamp(max=max_target)

                optimizer.zero_grad(set_to_none=True)

                # Compute log likelihoods and log priors
                likelihood_args, log_priors = model(data)
                preds = likelihood_args[..., 0]
                log_likelihoods = -torch.nn.functional.huber_loss(
                    preds,
                    target.expand_as(preds),
                    reduction="none",
                    delta=100,
                )

                # Re-weight log prior as in train_bbb.py
                loss = criterion(log_likelihoods, log_priors, 1 / len(train_loader))

                # Backpropagation
                loss.backward()

                train_loss += loss.item()

                # Precondition gradients
                preconditioner.step()

                # Update parameters
                optimizer.step()

            train_loss /= len(train_dataset)

            # Step learning rate scheduler
            lr_scheduler.step()

            # Save performance metrics
            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "num_parameters": num_parameters,
            }

            # For correct metric logging, fill expected
            # test metrics with None placeholders
            for test_max_target in test_max_targets:
                metrics.update(
                    {
                        f"prob_late_pred": None,
                        f"test_rmse_{test_max_target}": None,
                        f"test_mae_{test_max_target}": None,
                        f"test_score_{test_max_target}": None,
                        f"test_rmse_star_{test_max_target}": None,
                        f"test_mae_star_{test_max_target}": None,
                        f"test_score_star_{test_max_target}": None,
                        f"test_epistemic_uncertainty_{test_max_target}": None,
                    }
                )

            # Save checkpoint if specified
            checkpoint = None
            if (checkpoint_freq != 0 and epoch % checkpoint_freq == 0) or (
                checkpoint_at_end and epoch == num_epochs - 1
            ):
                checkpoint = {}
                checkpoint["rng_state"] = torch.random.get_rng_state()
                checkpoint["epoch"] = epoch
                checkpoint["model"] = model.state_dict()
                checkpoint["optimizer"] = optimizer.state_dict()
                checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
                checkpoint = Checkpoint.from_dict(checkpoint)

            if epoch < num_epochs - 1:
                # Report performance metrics
                session.report(metrics, checkpoint=checkpoint)

    # Build test dataset and dataloader
    test_dataset = CMAPSS(
        data_archive,
        subset=subset,
        split="test",
        window_size=window_size,
        normalization=normalization,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        pin_memory=device == "cuda",
    )

    # Test both without and with target rectification
    for test_max_target in test_max_targets:
        # Reseed for each test_max_target to make test deterministic
        torch.manual_seed(seed)

        prob_late_pred = 0.0

        # Estimate probability of late prediction on the train set
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            # Apply target rectification
            target = target.clamp(max=test_max_target)

            # For better performance, disable gradient
            # computation (unused at test time)
            with torch.no_grad():
                likelihood_args, _ = model(data)
            preds = likelihood_args[..., 0]
            prob_late_pred += (preds.mean(dim=0) > target).sum().item()

        prob_late_pred /= len(train_dataset)

        # Reseed for each test_max_target to make test deterministic
        torch.manual_seed(seed)

        # Test both without and with error reduction heuristic
        test_rmse, test_rmse_star = 0.0, 0.0
        test_mae, test_mae_star = 0.0, 0.0
        # Compute asymmetric score function as described in
        # https://ieeexplore.ieee.org/abstract/document/4711414
        test_score, test_score_star = 0.0, 0.0
        test_epistemic_uncertainty = 0.0
        model.eval()

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Apply target rectification
            target = target.clamp(max=test_max_target)

            # For better performance, disable gradient
            # computation (unused at test time)
            with torch.no_grad():
                likelihood_args, _ = model(data)
            preds = likelihood_args[..., 0]

            pred_mean = preds.mean(dim=0)
            pred_squared_mean = (preds**2).mean(dim=0)

            # Test without error reduction heuristic
            pred = pred_mean
            delta = pred - target
            test_rmse += (delta**2).sum().item()
            test_mae += delta.abs().sum().item()
            test_score += (
                (torch.where(delta < 0.0, -delta / 13.0, delta / 10.0).exp() - 1)
                .sum()
                .item()
            )

            # Test with error reduction heuristic
            pred_stddev = (pred_squared_mean - pred_mean**2).sqrt()
            pred_star = pred_mean - prob_late_pred * pred_stddev
            delta_star = pred_star - target
            test_rmse_star += (delta_star**2).sum().item()
            test_mae_star += delta_star.abs().sum().item()
            test_score_star += (
                (
                    torch.where(
                        delta_star < 0.0, -delta_star / 13.0, delta_star / 10.0
                    ).exp()
                    - 1
                )
                .sum()
                .item()
            )

            test_epistemic_uncertainty += (
                (pred_squared_mean - pred_mean**2).sum().item()
            )

            # Plot heatmap and posterior predictive
            if plot and test_max_target == float("inf"):
                if checkpoint is not None:
                    with checkpoint.as_directory():
                        os.makedirs("images", exist_ok=True)
                        for idx in range(50):
                            try:
                                plot_heatmap(
                                    data[idx, 0].cpu().numpy(),
                                    os.path.join(
                                        "images", f"data_{subset}_test_{idx}.pdf"
                                    ),
                                    usetex=True,
                                )
                                plot_distributions(
                                    {
                                        "": preds[:, idx : idx + 1].cpu().numpy(),
                                    },
                                    os.path.join(
                                        "images",
                                        f"prediction_{subset}_test_{idx}.pdf",
                                    ),
                                    vlines=[
                                        {
                                            "x": target[idx].cpu().item(),
                                            "linestyle": "--",
                                            "color": "k",
                                            "linewidth": 1,
                                        },
                                        {
                                            "x": pred[idx].cpu().item(),
                                            "linestyle": "--",
                                            "color": "g",
                                            "linewidth": 1,
                                        },
                                        {
                                            "x": pred_star[idx].cpu().item(),
                                            "linestyle": "--",
                                            "color": "r",
                                            "linewidth": 1,
                                        },
                                    ],
                                    xlabel="Remaining useful life",
                                    ylabel="Probability density",
                                    xlims=(
                                        target[idx].cpu().item() - 40,
                                        target[idx].cpu().item() + 40,
                                    ),
                                    usetex=True,
                                )
                            except Exception as e:
                                _LOGGER.warning(
                                    f"Could not plot heatmap and/or posterior predictive: {e}"
                                )

        test_rmse /= len(test_dataset)
        test_rmse = math.sqrt(test_rmse)
        test_mae /= len(test_dataset)
        test_rmse_star /= len(test_dataset)
        test_rmse_star = math.sqrt(test_rmse_star)
        test_mae_star /= len(test_dataset)
        test_epistemic_uncertainty /= len(test_dataset)

        # Save performance metrics
        metrics.update(
            {
                f"prob_late_pred": prob_late_pred,
                f"test_rmse_{test_max_target}": test_rmse,
                f"test_mae_{test_max_target}": test_mae,
                f"test_score_{test_max_target}": test_score,
                f"test_rmse_star_{test_max_target}": test_rmse_star,
                f"test_mae_star_{test_max_target}": test_mae_star,
                f"test_score_star_{test_max_target}": test_score_star,
                f"test_epistemic_uncertainty_{test_max_target}": test_epistemic_uncertainty,
            }
        )

    # Report performance metrics
    session.report(metrics, checkpoint=checkpoint)

    # Plot prior and posterior
    if checkpoint is not None:
        with checkpoint.as_directory():
            prior_samples = torch.normal(
                model.prior.mean[0].item(),
                model.prior.stddev[0].item(),
                size=(num_particles, 1),
            ).numpy()
            with torch.no_grad():
                particles = nn.utils.parameters_to_vector(model.parameters()).reshape(
                    num_particles, -1
                )
            posterior_samples = particles[:, :1].cpu().numpy()
            try:
                if plot:
                    os.makedirs("images", exist_ok=True)
                    plot_distributions(
                        {"": posterior_samples, None: prior_samples},
                        os.path.join("images", f"prior_posterior_epoch={epoch}.pdf"),
                        {None: "normal"},
                        xlabel=f"Weight \#1",
                        ylabel="Probability density",
                        xlims=(-2.5, 2.5),
                        usetex=True,
                    )
            except Exception as e:
                _LOGGER.warning(f"Could not plot prior/posterior: {e}")
            finally:
                del prior_samples, posterior_samples


if __name__ == "__main__":
    # Command-line argument parser
    parser = argparse.ArgumentParser("Run experiment")
    parser.add_argument(
        "model", type=str.lower, choices=["d3", "c2p2"], help="model to train"
    )
    parser.add_argument(
        "subset",
        type=str.lower,
        choices=["fd001", "fd002", "fd003", "fd004"],
        help="subset of C-MAPSS for training and testing the model",
    )
    parser.add_argument(
        "-r",
        "--restore",
        help="absolute or relative path to directory containing saved experiment state to restore",
    )
    args = parser.parse_args()

    # Build tuner
    experiment_state_dir = args.restore
    if experiment_state_dir is not None:
        tuner = Tuner.restore(path=experiment_state_dir)
    else:
        model_name = args.model
        subset = args.subset
        window_size = {"fd001": 30, "fd002": 20, "fd003": 30, "fd004": 15}[subset]
        local_dir = os.path.join(_ROOT_DIR, "experiments")
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"svgd_{model_name}_{subset}_{current_time}"
        tuner = Tuner(
            with_resources(
                run_experiment,
                {"cpu": 1, "gpu": torch.cuda.is_available()},
            ),
            param_space={
                "seed": grid_search(range(10)),
                "data_archive": os.path.join(_ROOT_DIR, "data", "cmapss.tar.gz"),
                "subset": subset,
                "window_size": window_size,
                "normalization": "min-max",
                "max_target": 125,
                "model_name": model_name,
                "num_epochs": 50,
                "batch_size": 512,
                "normal_prior_std": 0.1,
                "num_particles": 10,
                "lr": 1e-2,
                "step_size": 40,
                "gamma": 0.1,
                "checkpoint_freq": 10,
                "checkpoint_at_end": True,
                "test_only": False,
                "plot": False,
            },
            run_config=RunConfig(
                name=run_name,
                local_dir=local_dir,
                log_to_file=True,
            ),
        )

    # Run experiment
    analysis = tuner.fit().get_dataframe()
    run_dir = os.path.dirname(analysis["logdir"][0])
    num_seeds = len(analysis["config/seed"])
    metrics = analysis[
        [column for column in analysis.columns if column.startswith("test_")]
        + ["time_total_s"]
    ]
    mean, std = metrics.mean(axis=0), metrics.std(axis=0, ddof=0)
    with open(os.path.join(run_dir, "metrics.txt"), "w") as f:
        f.write(f"Metric values averaged over {num_seeds} random seeds:\n")
        f.write(
            "\n".join(
                [f"{name}: {mean[name]} +- {std[name]}" for name in metrics.keys()]
            )
        )
    with open(os.path.join(run_dir, "metrics.txt")) as f:
        print(f.read(), end="")
