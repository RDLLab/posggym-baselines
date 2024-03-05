"""Actor and Critic network for PPO algorithm."""
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    """Initialize a linear layer."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOModel(nn.Module):
    """Base PPO model class.

    Encapsulates the actor and critic networks.
    """

    def get_value(
        self,
        x: torch.tensor,
        lstm_state: Optional[Tuple[torch.tensor, torch.tensor]],
        done: torch.tensor,
    ) -> torch.tensor:
        """Get the value from the critic.

        B = batch_size
        M = num_lstm_layers
        L = lstm_size

        Arguments
        ---------
        x:
            The input to the network. Shape=(B, input_size).
        lstm_state:
            The previous hidden state of the LSTM (may be None if actor is not an RNN).
            If provided, Shape=(M, B, L).
        done:
            Whether the episode is done. Shape=(B,).
        action:
            The action taken. If None, sample from the actor. Shape=(B,).

        Returns
        -------
        value:
            The value of the state. Shape=(B,).

        """
        raise NotImplementedError

    def get_action(
        self,
        x: torch.tensor,
        lstm_state: Optional[Tuple[torch.tensor, torch.tensor]],
        done: torch.tensor,
        action: Optional[torch.tensor] = None,
    ) -> Tuple[
        torch.tensor,
        torch.tensor,
        torch.tensor,
        Optional[Tuple[torch.tensor, torch.tensor]],
    ]:
        """Get action from the actor.

        B = batch_size
        L = lstm_size
        M = num_lstm_layers

        Arguments
        ---------
        x:
            The input to the network. Shape (B, input_size).
        lstm_state:
            The previous hidden state of the LSTM (may be None if actor is not an RNN).
            If provided, Shape=(M, B, L).
        done:
            Whether the episode is done. Shape=(B,).
        action:
            The action taken. If None, sample from the actor. Shape=(B,).

        Returns
        -------
        action:
            The action. Shape=(B,).
        log_prob:
            The log probability of the action. Shape=(B,).
        entropy:
            The entropy of the action distribution. Shape=(B,).
        lstm_state:
            Optional, new state of the LSTM. Shape=(M, B, L).

        """
        raise NotImplementedError

    def get_action_and_value(
        self,
        x: torch.tensor,
        lstm_state: Optional[Tuple[torch.tensor, torch.tensor]],
        done: torch.tensor,
        action: Optional[torch.tensor] = None,
    ) -> Tuple[
        torch.tensor,
        torch.tensor,
        torch.tensor,
        torch.tensor,
        Optional[Tuple[torch.tensor, torch.tensor]],
    ]:
        """Get action from the actor and value from the critic.

        B = batch_size
        L = lstm_size
        M = num_lstm_layers

        Arguments
        ---------
        x:
            The input to the network. Shape (B, input_size).
        lstm_state:
            The previous hidden state of the LSTM (may be None if actor is not an RNN).
            If provided, Shape=(M, B, L).
        done:
            Whether the episode is done. Shape=(B,).
        action:
            The action taken. If None, sample from the actor. Shape=(B,).

        Returns
        -------
        action:
            The action. Shape=(B,).
        log_prob:
            The log probability of the action. Shape=(B,).
        entropy:
            The entropy of the action distribution. Shape=(B,).
        value:
            The values from critic. Shape=(B,).
        lstm_state:
            Optional, new state of the LSTM. Shape=(B, L).

        """
        raise NotImplementedError


class PPOLSTMModel(PPOModel):
    """PPO LSTM model class.

    Has linear trunk with tanh activations followed by an LSTM layer. The output
    of the LSTM layer is split into two heads, one for the actor (policy) and one for
    the (critic) value function.

    """

    def __init__(
        self,
        input_size: int,
        num_actions: int | List[int],
        trunk_sizes: List[int],
        lstm_size: int,
        lstm_layers: int,
        head_sizes: List[int],
        use_residual_lstm: bool,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_actions = num_actions
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.use_residual_lstm = use_residual_lstm

        prev_size = input_size
        trunk = []
        for size in trunk_sizes:
            trunk += [
                layer_init(nn.Linear(prev_size, size)),
                nn.Tanh(),
            ]
            prev_size = size
        self.trunk = nn.Sequential(*trunk)

        self.lstm = nn.LSTM(
            prev_size, lstm_size, num_layers=lstm_layers, batch_first=False
        )
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        prev_size = lstm_size

        actor = []
        critic = []
        for size in head_sizes:
            actor += [
                layer_init(nn.Linear(prev_size, size)),
                nn.Tanh(),
            ]
            critic += [
                layer_init(nn.Linear(prev_size, size)),
                nn.Tanh(),
            ]
            prev_size = size

        if isinstance(num_actions, list):
            actor.append(layer_init(nn.Linear(prev_size, sum(num_actions)), std=0.01))
        else:
            actor.append(layer_init(nn.Linear(prev_size, num_actions), std=0.01))

        self.actor = nn.Sequential(*actor)

        critic.append(
            layer_init(nn.Linear(prev_size, 1), std=1),
        )
        self.critic = nn.Sequential(*critic)

    def get_states(
        self,
        x: torch.tensor,
        lstm_state: Tuple[torch.tensor, torch.tensor],
        done: torch.tensor,
    ) -> Tuple[torch.tensor, Tuple[torch.tensor, torch.tensor]]:
        """Get the next states from the LSTM.

        B = batch_size (typically the number of parallel environments contained in the
            current batch, or the number of chunked sequences in the batch)
        T = seq_len (the length of each chunked sequence)
        L = lstm_size (the size of the LSTM hidden state)

        E.g. if each batch is collected from 8 parallel environments, and 128 steps
        are collected from each environment, and we chunk each sequence of 128 steps
        into 16 steps giving eight 16 step sequences per parallel environment, then
        `batch_size` is 8 * 8 = 64 and `seq_len` is 16.

        Arguments
        ---------
        x:
            The input to the network. Shape=(T * B, input_size).
        lstm_state:
            The previous state of the LSTM. This is a tuple with two entries, each of
            which has shape=(lstm.num_layers, B, L).
        done:
            Whether the episode is done at each step. Shape=(T * B,).

        Returns
        -------
        new_hidden:
            The output of the LSTM layer for each input x. Shape=(T * B, L)
        lstm_state:
            The new state of the LSTM at the end of the sequence. This is a
            tuple with two entries, each with shape=(lstm.num_layers, B, L).
        """
        hidden = self.trunk(x)

        # LSTM Logic
        # get hidden and lstm state for each timestep in the sequence
        # resets the hidden state when episode done
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]

        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        if self.use_residual_lstm:
            new_hidden += hidden.reshape(new_hidden.shape)
        return new_hidden, lstm_state

    def get_value(
        self,
        x: torch.tensor,
        lstm_state: Optional[Tuple[torch.tensor, torch.tensor]],
        done: torch.tensor,
    ) -> torch.tensor:
        assert lstm_state is not None
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action(
        self,
        x: torch.tensor,
        lstm_state: Optional[Tuple[torch.tensor, torch.tensor]],
        done: torch.tensor,
        action: Optional[torch.tensor] = None,
    ) -> Tuple[
        torch.tensor,
        torch.tensor,
        torch.tensor,
        Optional[Tuple[torch.tensor, torch.tensor]],
    ]:
        assert lstm_state is not None
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)

        def transpose(x):
            return x

        def transposeSum(x):
            return x

        if isinstance(self.num_actions, list):
            # We split this in to batches for running through categorical
            # One batch for each agent
            logits = logits.view(
                logits.shape[0], len(self.num_actions), self.num_actions[0]
            ).transpose(1, 0)

            def transpose(x):  # noqa: F811
                return x.T

            def transposeSum(x):  # noqa: F811
                return x.T.sum(1)

        probs = Categorical(logits=logits)

        action = probs.sample() if action is None else transpose(action)
        return (
            transpose(action),
            transposeSum(probs.log_prob(action)),
            transposeSum(probs.entropy()),
            lstm_state,
        )

    def get_action_and_value(
        self,
        x: torch.tensor,
        lstm_state: Optional[Tuple[torch.tensor, torch.tensor]],
        done: torch.tensor,
        action: Optional[torch.tensor] = None,
    ) -> Tuple[
        torch.tensor,
        torch.tensor,
        torch.tensor,
        torch.tensor,
        Optional[Tuple[torch.tensor, torch.tensor]],
    ]:
        assert lstm_state is not None
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)

        def transpose(x):
            return x

        def transposeSum(x):
            return x

        if isinstance(self.num_actions, list):
            # We split this in to batches for running through categorical
            # One batch for each agent
            logits = logits.view(
                logits.shape[0], len(self.num_actions), self.num_actions[0]
            ).transpose(1, 0)

            def transpose(x):  # noqa: F811
                return x.T

            def transposeSum(x):  # noqa: F811
                return x.T.sum(1)

        probs = Categorical(logits=logits)
        action = probs.sample() if action is None else transpose(action)

        return (
            transpose(action),
            transposeSum(probs.log_prob(action)),
            transposeSum(probs.entropy()),
            self.critic(hidden),
            lstm_state,
        )


class PPOMLPModel(PPOModel):
    """PPO MLP model class.

    Has linear trunk with tanh activations the output of which is split into two heads,
    one for the actor (policy) and one for the (critic) value function.

    """

    def __init__(
        self,
        input_size: int,
        num_actions: int,
        trunk_sizes: List[int],
        head_sizes: List[int],
    ):
        super().__init__()
        self.input_size = input_size
        self.num_actions = num_actions

        prev_size = input_size
        trunk = []
        for size in trunk_sizes:
            trunk += [
                layer_init(nn.Linear(prev_size, size)),
                nn.Tanh(),
            ]
            prev_size = size
        self.trunk = nn.Sequential(*trunk)

        actor = []
        critic = []
        for size in head_sizes:
            actor += [
                layer_init(nn.Linear(prev_size, size)),
                nn.Tanh(),
            ]
            critic += [
                layer_init(nn.Linear(prev_size, size)),
                nn.Tanh(),
            ]
            prev_size = size

        actor.append(layer_init(nn.Linear(prev_size, num_actions), std=0.01))
        self.actor = nn.Sequential(*actor)

        critic.append(
            layer_init(nn.Linear(prev_size, 1), std=1),
        )
        self.critic = nn.Sequential(*critic)

    def get_value(
        self,
        x: torch.tensor,
        lstm_state: Optional[Tuple[torch.tensor, torch.tensor]],
        done: torch.tensor,
    ) -> torch.tensor:
        hidden = self.trunk(x)
        return self.critic(hidden)

    def get_action(
        self,
        x: torch.tensor,
        lstm_state: Optional[Tuple[torch.tensor, torch.tensor]],
        done: torch.tensor,
        action: Optional[torch.tensor] = None,
    ) -> Tuple[
        torch.tensor,
        torch.tensor,
        torch.tensor,
        Optional[Tuple[torch.tensor, torch.tensor]],
    ]:
        hidden = self.trunk(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            lstm_state,
        )

    def get_action_and_value(
        self,
        x: torch.tensor,
        lstm_state: Optional[Tuple[torch.tensor, torch.tensor]],
        done: torch.tensor,
        action: Optional[torch.tensor] = None,
    ) -> Tuple[
        torch.tensor,
        torch.tensor,
        torch.tensor,
        torch.tensor,
        Optional[Tuple[torch.tensor, torch.tensor]],
    ]:
        hidden = self.trunk(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(hidden),
            lstm_state,
        )
