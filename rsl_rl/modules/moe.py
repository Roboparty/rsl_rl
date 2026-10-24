# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.utils import resolve_nn_activation


class MoeLayer(nn.Module):
    """Soft mixture-of-experts: gate mixes expert MLP outputs."""

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        output_dim: int,
        *,
        activation: str = "elu",
        expert_hidden_dims: list[int] | None = None,
        gate_hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        expert_hidden_dims = list(expert_hidden_dims or [])
        gate_hidden_dims = list(gate_hidden_dims or [])
        self._activation_name = activation
        self.gate = self._build_gate(input_dim, num_experts, gate_hidden_dims)
        self.experts = nn.ModuleList(
            [
                self._build_expert(input_dim, output_dim, expert_hidden_dims)
                for _ in range(num_experts)
            ]
        )

    def _build_gate(self, input_dim: int, num_experts: int, hidden_dims: list[int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(resolve_nn_activation(self._activation_name))
            curr_dim = h
        layers.append(nn.Linear(curr_dim, num_experts))
        return nn.Sequential(*layers)

    def _build_expert(self, input_dim: int, output_dim: int, hidden_dims: list[int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(resolve_nn_activation(self._activation_name))
            curr_dim = h
        layers.append(nn.Linear(curr_dim, output_dim))
        return nn.Sequential(*layers)

    def gate_scores(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.gate(x), dim=-1)

    @torch.no_grad()
    def gate_stats(self, x: torch.Tensor) -> dict[str, float]:
        gate_scores = self.gate_scores(x)
        mean_weights = gate_scores.mean(dim=0)
        entropy = -(gate_scores * (gate_scores + 1e-8).log()).sum(dim=-1).mean()
        stats = {f"expert_{i}": mean_weights[i].item() for i in range(gate_scores.shape[-1])}
        stats["gate_entropy"] = entropy.item()
        stats["max_weight"] = gate_scores.max(dim=-1).values.mean().item()
        return stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_scores = self.gate_scores(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        return torch.einsum("be,beo->bo", gate_scores, expert_outputs)
