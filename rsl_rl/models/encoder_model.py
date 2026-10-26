# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Parkour image encoders and mixture-of-experts heads built on the standard CNN model."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.cnn_model import CNNModel
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.modules.moe import MoeLayer
from rsl_rl.utils import resolve_nn_activation


class Conv2dHeadModel(nn.Module):
    """The existing Parkour encoder architecture, with a 5.x model interface."""

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        channels: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        hidden_sizes: list[int],
        output_size: int,
        paddings: list[int] | None = None,
        nonlinearity: str = "ReLU",
        use_maxpool: bool = False,
        last_activation: str | None = None,
        input_dtype: str | None = None,
        input_range: tuple[float, float] = (0.0, 255.0),
    ) -> None:
        super().__init__()
        self.input_dtype = input_dtype
        self.input_range = tuple(input_range)
        if input_dtype not in (None, "uint8", "float32"):
            raise ValueError("Encoder input_dtype must be uint8 or float32.")
        if len(input_range) != 2 or input_range[0] >= input_range[1]:
            raise ValueError("Encoder input_range must be increasing.")
        if input_dtype == "uint8" and self.input_range != (0.0, 255.0):
            raise ValueError("uint8 depth inputs must declare input_range=(0, 255).")
        paddings = [0] * len(channels) if paddings is None else paddings
        if not len(channels) == len(kernel_sizes) == len(strides) == len(paddings):
            raise ValueError("Encoder channels, kernels, strides and paddings must have the same length.")
        layers = []
        in_channels = [image_shape[0], *channels[:-1]]
        for in_ch, out_ch, kernel, stride, padding in zip(in_channels, channels, kernel_sizes, strides, paddings):
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel, 1 if use_maxpool else stride, padding),
                    resolve_nn_activation(nonlinearity),
                ]
            )
            if use_maxpool and stride > 1:
                layers.append(nn.MaxPool2d(stride))
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            conv_dim = self.conv(torch.zeros(1, *image_shape)).numel()
        self.head = MLP(conv_dim, output_size, hidden_sizes, nonlinearity, last_activation=last_activation)
        self.out_features = output_size
        self.output_dim = output_size
        self.output_channels = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_dtype == "uint8" and x.dtype != torch.uint8:
            raise TypeError("Depth encoder expects uint8 input in [0, 255].")
        if self.input_dtype == "float32":
            if x.dtype != torch.float32:
                raise TypeError("Depth encoder expects float32 input.")
            x = (x - self.input_range[0]) / (self.input_range[1] - self.input_range[0])
        if x.dtype == torch.uint8:
            x = x.float().div(255.0)
        return self.head(self.conv(x).flatten(start_dim=1))


class EncoderModel(CNNModel):
    """Use the original Parkour encoder with the standard observation and model interfaces."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        encoder_obs_groups: tuple[str, ...] | list[str] = ("depth_image",),
        encoder_cfg: dict | None = None,
    ) -> None:
        if encoder_cfg is None or "output_size" not in encoder_cfg:
            raise ValueError("EncoderModel requires encoder_cfg.output_size.")
        cnns = {
            group: Conv2dHeadModel(tuple(obs[group].shape[1:]), **encoder_cfg)
            for group in obs_groups[obs_set]
            if group in encoder_obs_groups
        }
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
            cnns=cnns,
        )


class EncoderMoEModel(EncoderModel):
    """Parkour model with soft mixture-of-experts in the actor or critic head."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        num_moe_experts: int = 8,
        moe_gate_hidden_dims: list[int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg, **kwargs
        )
        if self.distribution is not None and self.distribution.input_dim != self.distribution.output_dim:
            raise NotImplementedError("EncoderMoEModel requires a state-independent output distribution.")
        output_dim = self.mlp[-1].out_features
        self.mlp = MoeLayer(
            self._get_latent_dim(),
            num_moe_experts,
            output_dim,
            activation=activation,
            expert_hidden_dims=list(hidden_dims),
            gate_hidden_dims=list(moe_gate_hidden_dims or []),
        )
