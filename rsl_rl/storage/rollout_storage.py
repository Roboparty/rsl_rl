# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict

from rsl_rl.modules import HiddenState
from rsl_rl.utils import split_and_pad_trajectories


class RolloutStorage:
    """Storage for the data collected during a rollout.

    The rollout storage is populated by adding transitions during the rollout phase. It then returns a generator for
    learning, depending on the algorithm and the policy architecture.
    """

    class Transition:
        """Storage for a single state transition.

        This class is populated incrementally during the rollout phase and then passed to
        :meth:`RolloutStorage.add_transition` to record the data.
        """

        def __init__(self) -> None:
            """Initialize an empty transition container."""
            self.observations: TensorDict | None = None
            """Observations at the current step."""

            self.actions: torch.Tensor | None = None
            """Actions taken at the current step."""

            self.rewards: torch.Tensor | None = None
            """Rewards received after the action."""

            self.dones: torch.Tensor | None = None
            """Done flags indicating episode termination."""

            # For reinforcement learning
            self.values: torch.Tensor | None = None
            """Value estimates at the current step (RL only)."""

            self.actions_log_prob: torch.Tensor | None = None
            """Log probability of the taken actions (RL only)."""

            self.distribution_params: tuple[torch.Tensor, ...] | None = None
            """Parameters of the action distribution (RL only)."""

            # For distillation
            self.privileged_actions: torch.Tensor | None = None
            """Privileged (teacher) actions (distillation only)."""

            # For recurrent networks
            self.hidden_states: tuple[HiddenState, HiddenState] = (None, None)
            """Hidden states for recurrent networks, e.g., (actor, critic)."""

        def clear(self) -> None:
            """Reset all transition fields to None."""
            self.__init__()

    class Batch:
        """A batch of data yielded by the rollout storage generators.

        This class provides named access to mini-batch fields. Fields are optional to support different training modes
        (RL vs distillation) and architectures (feedforward vs recurrent).
        """

        def __init__(
            self,
            observations: TensorDict | None = None,
            actions: torch.Tensor | None = None,
            values: torch.Tensor | None = None,
            advantages: torch.Tensor | None = None,
            returns: torch.Tensor | None = None,
            old_actions_log_prob: torch.Tensor | None = None,
            old_distribution_params: tuple[torch.Tensor, ...] | None = None,
            hidden_states: tuple[HiddenState, HiddenState] = (None, None),
            masks: torch.Tensor | None = None,
            privileged_actions: torch.Tensor | None = None,
            dones: torch.Tensor | None = None,
        ) -> None:
            """Initialize a batch container over rollout data."""
            self.observations: TensorDict | None = observations
            """Batch of observations."""

            # For reinforcement learning
            self.actions: torch.Tensor | None = actions
            """Batch of actions."""

            self.values: torch.Tensor | None = values
            """Batch of value estimates (RL only)."""

            self.advantages: torch.Tensor | None = advantages
            """Batch of advantage estimates (RL only)."""

            self.returns: torch.Tensor | None = returns
            """Batch of return targets (RL only)."""

            self.old_actions_log_prob: torch.Tensor | None = old_actions_log_prob
            """Batch of log probabilities of the old actions (RL only)."""

            self.old_distribution_params: tuple[torch.Tensor, ...] | None = old_distribution_params
            """Batch of parameters of the old action distribution (RL only)."""

            # For distillation
            self.privileged_actions: torch.Tensor | None = privileged_actions
            """Batch of privileged (teacher) actions (distillation only)."""

            self.dones: torch.Tensor | None = dones
            """Batch of done flags (distillation only)."""

            # For recurrent networks
            self.hidden_states: tuple[HiddenState, HiddenState] = hidden_states
            """Batch of hidden states for recurrent networks (RL recurrent only)."""

            self.masks: torch.Tensor | None = masks
            """Batch of trajectory masks for recurrent networks (RL recurrent only)."""

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
        compact_depth_history: bool = False,
        depth_history_offsets: tuple[int, ...] | list[int] = (),
        depth_observation_key: tuple[str, ...] = ("depth",),
    ) -> None:
        """Allocate rollout buffers for a specific training mode and batch shape."""
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.actions_shape = actions_shape
        self.compact_depth_history = compact_depth_history
        self.depth_history_offsets = tuple(depth_history_offsets) if compact_depth_history else ()
        self.depth_observation_key = tuple(depth_observation_key)

        if compact_depth_history:
            offsets = self.depth_history_offsets
            if (
                not offsets
                or offsets[-1] != 0
                or any(type(offset) is not int or offset < 0 for offset in offsets)
                or any(older <= newer for older, newer in zip(offsets, offsets[1:]))
            ):
                raise ValueError("Depth history offsets must be descending nonnegative integers ending in zero")
            depth = obs.get(self.depth_observation_key)
            if depth is None:
                raise KeyError(f"Missing compact depth observation at {self.depth_observation_key!r}")
            if depth.ndim != 4 or depth.shape[0] != num_envs or depth.shape[1] != len(offsets):
                raise ValueError(
                    "Compact rollout requires depth [env, frames, height, width] matching the configured offsets"
                )
            if depth.dtype != torch.uint8:
                raise ValueError("Compact rollout requires uint8 depth observations")

        # Core
        self.observations = TensorDict(
            {
                key: torch.zeros(num_transitions_per_env, *value.shape, dtype=value.dtype, device=device)
                for key, value in obs.items()
            },
            batch_size=[num_transitions_per_env, num_envs],
            device=self.device,
        )
        if self.compact_depth_history:
            depth = obs.get(self.depth_observation_key)
            compact_shape = (num_transitions_per_env, num_envs, *depth.shape[2:])
            self.observations.set(
                self.depth_observation_key,
                torch.zeros(compact_shape, dtype=depth.dtype, device=device),
            )
            max_offset = self.depth_history_offsets[0]
            self._depth_offsets = torch.tensor(self.depth_history_offsets, device=device)
            self._depth_lookback = torch.arange(max_offset + 1, device=device)
            prefix_shape = (max_offset, num_envs, *depth.shape[2:])
            self._depth_prefix = torch.zeros(prefix_shape, dtype=depth.dtype, device=device)
            if max_offset:
                self._depth_prefix.copy_(depth[:, -1].unsqueeze(0).expand_as(self._depth_prefix))
            self._depth_episode_start = torch.zeros(
                num_transitions_per_env, num_envs, dtype=torch.bool, device=device
            )
            self._depth_start_prefix = torch.zeros(max_offset, num_envs, dtype=torch.bool, device=device)
            self._next_depth_episode_start = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For distillation
        if training_type == "distillation":
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # For reinforcement learning
        if training_type == "rl":
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.distribution_params: tuple[torch.Tensor, ...] | None = None  # Lazily initialized on first transition
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # For recurrent networks
        self.saved_hidden_state_a = None
        self.saved_hidden_state_c = None

        # Counter for the number of transitions stored
        self.step = 0

    def add_transition(self, transition: Transition) -> None:
        """Add one transition to the storage at the current step index."""
        # Check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        if self.compact_depth_history:
            self._copy_compact_observations(self.observations[self.step], transition.observations)
            self._depth_episode_start[self.step].copy_(self._next_depth_episode_start)
            self._next_depth_episode_start.copy_(transition.dones.reshape(self.num_envs, -1).bool().any(dim=-1))
        else:
            self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)  # type: ignore
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # For distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)  # type: ignore

        # For reinforcement learning
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)  # type: ignore
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            if self.distribution_params is None:  # Initialize the distribution parameters
                self.distribution_params = tuple(
                    torch.zeros(self.num_transitions_per_env, *p.shape, device=self.device)
                    for p in transition.distribution_params  # type: ignore
                )
            for i, p in enumerate(transition.distribution_params):  # type: ignore
                self.distribution_params[i][self.step].copy_(p)

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # Increment the counter
        self.step += 1

    def clear(self) -> None:
        """Reset the write cursor for the next rollout."""
        if self.compact_depth_history and self.depth_history_offsets[0]:
            depth = self.observations.get(self.depth_observation_key)[: self.step]
            starts = self._depth_episode_start[: self.step]
            prefix_length = len(self._depth_prefix)
            if self.step >= prefix_length:
                self._depth_prefix.copy_(depth[-prefix_length:])
                self._depth_start_prefix.copy_(starts[-prefix_length:])
            elif self.step:
                self._depth_prefix[:-self.step].copy_(self._depth_prefix[self.step:].clone())
                self._depth_prefix[-self.step:].copy_(depth)
                self._depth_start_prefix[:-self.step].copy_(self._depth_start_prefix[self.step:].clone())
                self._depth_start_prefix[-self.step:].copy_(starts)
        self.step = 0

    # For distillation
    def generator(self) -> Generator[Batch, None, None]:
        """Yield per-timestep batches for distillation training."""
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            observations = self.observations[i].clone()
            if self.compact_depth_history:
                time_idx = torch.full((self.num_envs,), i, device=self.device, dtype=torch.long)
                env_idx = torch.arange(self.num_envs, device=self.device)
                observations = self._restore_depth(observations, time_idx, env_idx)
            yield RolloutStorage.Batch(
                observations=observations,  # type: ignore
                privileged_actions=self.privileged_actions[i],
                dones=self.dones[i],
            )

    # For reinforcement learning with feedforward networks
    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator[Batch, None, None]:
        """Yield shuffled flat mini-batches for feedforward RL updates."""
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Flatten the data
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params)  # type: ignore
        depth_starts = (
            torch.cat((self._depth_start_prefix, self._depth_episode_start), dim=0)
            if self.compact_depth_history
            else None
        )

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                # Yield the mini-batch
                batch_observations = observations[batch_idx]
                if self.compact_depth_history:
                    time_idx = torch.div(batch_idx, self.num_envs, rounding_mode="floor")
                    env_idx = batch_idx % self.num_envs
                    batch_observations = self._restore_depth(
                        batch_observations, time_idx, env_idx, starts=depth_starts
                    )

                yield RolloutStorage.Batch(
                    observations=batch_observations,  # type: ignore
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                )

    # For reinforcement learning with recurrent networks
    def recurrent_mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[Batch, None, None]:
        """Yield trajectory mini-batches with masks and recurrent hidden states."""
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        observations = self._restore_all_depth() if self.compact_depth_history else self.observations
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(observations, self.dones)
        mini_batch_size = self.num_envs // num_mini_batches

        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                # Handle the hidden states
                # Reshape to [num_envs, time, num layers, hidden dim]
                # Original shape: [time, num_layers, num_envs, hidden_dim])
                last_was_done = last_was_done.permute(1, 0)
                # Take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                if self.saved_hidden_state_a is not None:
                    hidden_state_a_batch = [
                        saved_hidden_state.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                        .transpose(1, 0)
                        .contiguous()
                        for saved_hidden_state in self.saved_hidden_state_a
                    ]
                    # Unwrap the single state of a GRU, keep the pair of an LSTM as a tuple
                    hidden_state_a_batch = (
                        hidden_state_a_batch[0] if len(hidden_state_a_batch) == 1 else tuple(hidden_state_a_batch)
                    )
                else:
                    hidden_state_a_batch = None
                if self.saved_hidden_state_c is not None:
                    hidden_state_c_batch = [
                        saved_hidden_state.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                        .transpose(1, 0)
                        .contiguous()
                        for saved_hidden_state in self.saved_hidden_state_c
                    ]
                    hidden_state_c_batch = (
                        hidden_state_c_batch[0] if len(hidden_state_c_batch) == 1 else tuple(hidden_state_c_batch)
                    )
                else:
                    hidden_state_c_batch = None

                # Yield the mini-batch
                yield RolloutStorage.Batch(
                    observations=padded_obs_trajectories[:, first_traj:last_traj],  # type: ignore
                    actions=self.actions[:, start:stop],
                    values=self.values[:, start:stop],
                    advantages=self.advantages[:, start:stop],
                    returns=self.returns[:, start:stop],
                    old_actions_log_prob=self.actions_log_prob[:, start:stop],
                    old_distribution_params=tuple(p[:, start:stop] for p in self.distribution_params),  # type: ignore
                    hidden_states=(hidden_state_a_batch, hidden_state_c_batch),  # type: ignore
                    masks=trajectory_masks[:, first_traj:last_traj],
                )

                first_traj = last_traj


    def _copy_compact_observations(
        self, destination: TensorDict, source: TensorDict, prefix: tuple[str, ...] = ()
    ) -> None:
        """Copy observations while retaining only the newest depth frame."""
        for key, source_value in source.items():
            path = (*prefix, key)
            destination_value = destination.get(key)
            if isinstance(source_value, TensorDict):
                self._copy_compact_observations(destination_value, source_value, path)
            elif path == self.depth_observation_key:
                if source_value.ndim != 4 or source_value.shape[1] != len(self.depth_history_offsets):
                    raise ValueError(
                        "Compact rollout requires depth [env, frames, height, width] matching the configured offsets"
                    )
                if source_value.dtype != torch.uint8:
                    raise ValueError("Compact rollout requires uint8 depth observations")
                destination_value.copy_(source_value[:, -1])
            else:
                destination_value.copy_(source_value)

    def _restore_depth(
        self,
        observations: TensorDict,
        time_idx: torch.Tensor,
        env_idx: torch.Tensor,
        *,
        starts: torch.Tensor | None = None,
    ) -> TensorDict:
        """Rebuild depth stacks for selected rollout time/environment pairs."""
        max_offset = self.depth_history_offsets[0]
        depth = self.observations.get(self.depth_observation_key)
        if starts is None:
            starts = torch.cat((self._depth_start_prefix, self._depth_episode_start), dim=0)
        base = time_idx + max_offset
        lookback = self._depth_lookback
        history_starts = starts[base[:, None] - lookback, env_idx[:, None]]
        # Camera history repeats the first post-reset frame instead of crossing an episode boundary.
        nearest_start = torch.where(history_starts, lookback, max_offset).amin(dim=1)
        source = base[:, None] - torch.minimum(self._depth_offsets, nearest_start[:, None])
        source_env = env_idx[:, None].expand_as(source)
        prefix_mask = source < max_offset
        restored_depth = torch.empty(
            *source.shape, *depth.shape[2:], dtype=depth.dtype, device=depth.device
        )
        restored_depth[prefix_mask] = self._depth_prefix[source[prefix_mask], source_env[prefix_mask]]
        current_mask = ~prefix_mask
        restored_depth[current_mask] = depth[
            source[current_mask] - max_offset, source_env[current_mask]
        ]
        observations.set(self.depth_observation_key, restored_depth)
        return observations

    def _restore_all_depth(self) -> TensorDict:
        """Rebuild the full rollout depth tensor for recurrent trajectory splitting."""
        time_idx = torch.arange(self.num_transitions_per_env, device=self.device).repeat_interleave(self.num_envs)
        env_idx = torch.arange(self.num_envs, device=self.device).repeat(self.num_transitions_per_env)
        observations = self.observations.clone()
        flat_observations = self.observations.flatten(0, 1)[torch.arange(len(time_idx), device=self.device)]
        restored = self._restore_depth(flat_observations, time_idx, env_idx).get(self.depth_observation_key)
        observations.set(
            self.depth_observation_key,
            restored.reshape(self.num_transitions_per_env, self.num_envs, *restored.shape[1:]),
        )
        return observations

    def _save_hidden_states(self, hidden_states: tuple[HiddenState, HiddenState]) -> None:
        """Save recurrent hidden states to the rollout storage."""
        if hidden_states == (None, None):
            return
        # Make a tuple out of GRU hidden states to match the LSTM format
        if hidden_states[0] is not None:
            hidden_state_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        if hidden_states[1] is not None:
            hidden_state_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        # Initialize hidden states if needed
        if self.saved_hidden_state_a is None and hidden_states[0] is not None:
            self.saved_hidden_state_a = [
                torch.zeros(self.observations.shape[0], *hidden_state_a[i].shape, device=self.device)
                for i in range(len(hidden_state_a))
            ]
        if self.saved_hidden_state_c is None and hidden_states[1] is not None:
            self.saved_hidden_state_c = [
                torch.zeros(self.observations.shape[0], *hidden_state_c[i].shape, device=self.device)
                for i in range(len(hidden_state_c))
            ]
        # Copy the states
        if hidden_states[0] is not None:
            for i in range(len(hidden_state_a)):
                self.saved_hidden_state_a[i][self.step].copy_(hidden_state_a[i])  # type: ignore
        if hidden_states[1] is not None:
            for i in range(len(hidden_state_c)):
                self.saved_hidden_state_c[i][self.step].copy_(hidden_state_c[i])  # type: ignore
