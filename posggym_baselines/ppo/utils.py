"""Utility functions for PPO."""
from typing import TYPE_CHECKING, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


if TYPE_CHECKING:
    from posggym_baselines.ppo.config import PPOConfig


def split_batch_by_policy(
    batch: Dict[str, torch.tensor], config: "PPOConfig"
) -> Dict[str, Dict[str, torch.tensor]]:
    """Split a batch into batches for each policy.

    L = seq_len (max length for each individual sequence)
    B_j = batch_size for policy j (total number of sequences generated for policy j)
    B = batch_size (total number of sequences across all policies)
    N = number of agents in the environment

    Arguments
    ---------
    batch:
        The batch to split. A dictionary mapping a batch key (e.g. "obs", "actions",
        "logprobs", etc.) to a tensor. Most tensors will have shape=(L, B, N, ...),
        with some (e.g. lstm_states) having slightly different shapes.
    config:
        The configuration information.

    Returns
    -------
    policy_batches:
        A dictionary mapping policy ids to batches. Each batch is a dictionary mapping
        a batch key (e.g. "obs", "actions", "logprobs", etc.) to a tensor. Each tensor
        has shape like shape=(L, B_j, ...).
    """
    if config.num_agents == 1:
        partner_policy_idxs = None
    else:
        # create tensor of partner policy ids for each sequence
        # partner idxs to have extra dimension, shape=(L, B, N, N-1)
        partner_policy_idxs = torch.zeros(
            batch["policy_idxs"].shape + (config.num_agents - 1,)
        ).long()
        for i in range(config.num_agents):
            partner_policy_idxs[:, :, i] = torch.index_select(
                batch["policy_idxs"],
                dim=2,
                index=torch.tensor([j for j in range(config.num_agents) if j != i]),
            )

    policy_batches = {}
    for policy_id in config.get_all_policy_ids():
        policy_idx = config.get_policy_idx(policy_id)
        b_idxs = batch["policy_idxs"][0, :, :] == policy_idx
        policy_batch = {}
        for k, b in batch.items():
            if k == "initial_lstm_states":
                policy_batch[k] = (b[0][:, b_idxs], b[1][:, b_idxs])
            elif k == "policy_stats":
                # already split by policy in worker
                policy_batch[k] = b.get(policy_id, {})
            else:
                policy_batch[k] = b[:, b_idxs]
        if partner_policy_idxs is not None:
            policy_batch["partner_policy_idxs"] = partner_policy_idxs[:, b_idxs]
        else:
            policy_batch["partner_policy_idxs"] = None
        policy_batches[policy_id] = policy_batch
    return policy_batches


def filter_batch_by_partner(
    batch: Dict[str, torch.tensor], exclude_policy_ids: List[str], config: "PPOConfig"
) -> Dict[str, torch.tensor]:
    """Filter policy batch to exclude experience against certain partners.

    Arguments
    ---------
    batch:
        The batch to filter. A dictionary mapping a batch key (e.g. "obs", "actions",
        "logprobs", etc.) to a tensor. Most tensors will have shape=(L, B, N, ...),
        with some (e.g. lstm_states) having slightly different shapes.
    exclude_policy_ids:
        A list of policy ids to exclude.
    config:
        The configuration information.

    Returns
    -------
    filtered_batch:
        The batch with experience against given partners removed. If all experience
        is removed (i.e. all data in batch is with exluded partners), an empty
        dictionary will be returned.

    """
    if config.num_agents == 1:
        return batch
    assert "partner_policy_idxs" in batch
    # keep_idxs = torch.ones_like(batch["policy_idxs"])
    # for policy_id in exclude_policy_ids:
    #     policy_idx = config.get_policy_idx(policy_id)
    #     keep_idxs[batch["partner_policy_idxs"] == policy_idx] = 0
    # b_idxs = keep_idxs[0] > 0

    exclude_policy_idxs = torch.tensor(
        [config.get_policy_idx(policy_id) for policy_id in exclude_policy_ids]
    )
    # keep sequences where ALL of the partner policy idxs are NOT IN the exclude list
    # [0] is used here to select first entry of each sequence (since policy ID doesn't
    # change within a sequence) shape=(B, N-1)
    keep_idxs = torch.all(
        torch.isin(batch["partner_policy_idxs"][0], exclude_policy_idxs, invert=True),
        dim=-1,
    )

    filtered_batch = {}
    for k, b in batch.items():
        if k == "initial_lstm_states":
            filtered_batch[k] = (b[0][:, keep_idxs], b[1][:, keep_idxs])
        elif k == "policy_stats":
            # already split by policy in worker, nothing we can do here
            # but this won't be used in training anyway
            filtered_batch[k] = b
        else:
            filtered_batch[k] = b[:, keep_idxs]
    return filtered_batch


def filter_policy_batches_by_partner_dist(
    policy_batches: Dict[str, Dict[str, torch.tensor]], config: "PPOConfig"
) -> Dict[str, Dict[str, torch.tensor]]:
    """Filter policy batches to exclude experience against certain partners.

    Filtering is done according to the partner distribution for each policy given
    by `config.get_policy_partner_distribution`.

    Arguments
    ---------
    policy_batches:
        A dictionary mapping policy ids to batches.
    config:
        The configuration information.

    Returns
    -------
    filtered_policy_batches:
        The policy batches with experience against given partners removed. If all
        experience is removed (i.e. all data in batch is with exluded partners), an
        empty dictionary will be returned.

    """
    filtered_policy_batches = {}
    all_policy_ids = config.get_all_policy_ids()
    for policy_id, batch in policy_batches.items():
        dist = config.get_policy_partner_distribution(policy_id)
        exclude_policy_ids = [
            partner_id
            for partner_id in all_policy_ids
            if dist.get(partner_id, 0.0) == 0.0
        ]
        filtered_policy_batches[policy_id] = filter_batch_by_partner(
            batch, exclude_policy_ids, config
        )
    return filtered_policy_batches


def combine_batches(
    batches: List[Dict[str, torch.tensor]], config: "PPOConfig"
) -> Dict[str, Dict[str, torch.tensor]]:
    """Combine multiple batches into a single batch.

    L = seq_len (max length for each individual sequence)
    B_i = batch_size for batch i (total number of sequences in batch i)
    B = batch_size (total number of sequences across all batches)
    N = number of agents in the environment

    Arguments
    ---------
    batches:
        The list of batches to combine. Each batch is a dictionary mapping a batch key
        (e.g. "obs", "actions", "logprobs", etc.) to a tensor. Most tensors will have
        shape=(L, B_i, N, ...), with some (e.g. lstm_states) having slightly different
        shapes.
    config:
        The configuration information.

    Returns
    -------
    combined_batches:
        A dictionary mapping batch keys to tensors. Each tensor has shape like
        shape=(L, B, N, ...).

    """
    combined_batch = {}
    for k in batches[0]:
        if k == "initial_lstm_states":
            combined_batch[k] = (
                torch.cat([b[k][0] for b in batches], dim=1),
                torch.cat([b[k][1] for b in batches], dim=1),
            )
        elif k == "policy_stats":
            policy_stat_policy_ids = set()
            policy_stat_keys = set()
            for b in batches:
                policy_stat_policy_ids.update(b[k].keys())
                for policy_id, policy_stats in b[k].items():
                    policy_stat_keys.update(policy_stats.keys())

            combined_batch[k] = {}
            for policy_id in policy_stat_policy_ids:
                policy_stats = {}
                for stat_name in policy_stat_keys:
                    values = [
                        b[k][policy_id][stat_name]
                        for b in batches
                        if (policy_id in b[k] and stat_name in b[k][policy_id])
                    ]
                    if len(values) == 0:
                        continue
                    values = torch.tensor(values)
                    if "mean_" in stat_name:
                        policy_stats[stat_name] = torch.mean(values)
                    elif "min_" in stat_name:
                        policy_stats[stat_name] = torch.min(values)
                    elif "max_" in stat_name:
                        policy_stats[stat_name] = torch.max(values)
                    else:
                        # just take value from first worker
                        policy_stats[stat_name] = values[0]
                if len(policy_stats) > 0:
                    combined_batch[k][policy_id] = policy_stats
        else:
            combined_batch[k] = torch.cat([b[k] for b in batches], dim=1)

    return combined_batch


def get_seq_idxs(
    dones_buf: torch.tensor, max_seq_len: int
) -> Tuple[List[List[torch.tensor]], List[List[torch.tensor]]]:
    """Get sequence chunk indices for each batch dim.

    The sequence chunks are used to split the batch into sequences of length
    `config.seq_len` which can then be used to train the policy with BPTT.

    T = batch steps (i.e. config.num_rollout_steps)
    B = batch size (i.e. config.num_envs)

    Arguments
    ---------
    dones_buf:
        The dones buffer, storing whether the prev step in the sequence was a terminal
        state or not. Shape=(T, B).
    max_seq_len:
        The max sequence chunk length.

    Returns
    -------
    seq_idxs:
        The indices of each sequence chunk for each batch dim. The number of chunks for
        each batch dim may vary (i.e. due to episodes terminated at different steps,
        etc.) Shape=(B, num_seqs_chunks along batch dim).
    overlapping_seq_idxs:
        Same as `seq_idxs`, but each chunk includes the first idx of the next chunk.
        This is used for calculating the advantages and returns, since the value of the
        next state is used in the calculation.

    """
    num_rollout_steps, num_envs = dones_buf.shape
    seq_idxs = []
    overlapping_seq_idxs = []
    for env_idx in range(num_envs):
        env_done_idxs = torch.nonzero(dones_buf[:, env_idx]).squeeze(1)
        env_episode_idxs = torch.tensor_split(
            torch.arange(num_rollout_steps), env_done_idxs
        )
        env_seq_idxs = []
        for episode_idxs in env_episode_idxs:
            if len(episode_idxs) > 0:
                env_seq_idxs.extend(torch.split(episode_idxs, max_seq_len))
        overapping_env_seq_idxs = []
        for i in range(len(env_seq_idxs)):
            next_idx = env_seq_idxs[i][-1] + 1
            overapping_env_seq_idxs.append(
                torch.cat((env_seq_idxs[i], torch.tensor([next_idx]).long()))
            )

        seq_idxs.append(env_seq_idxs)
        overlapping_seq_idxs.append(overapping_env_seq_idxs)
    return seq_idxs, overlapping_seq_idxs


def split_and_pad_batch(
    batch: torch.tensor,
    seq_idxs: List[List[torch.tensor]],
    max_seq_len: int,
    padding_value: float = 0.0,
) -> torch.tensor:
    """Split batch into sequences and pad to max_seq_len.

    The batch is first split into episodes, then each episode is split into sequences
    of length `max_seq_len`. Any episode sequences that are shorter than `max_seq_len`
    are zero padded to `max_seq_len`.

    T = batch steps (i.e. config.num_rollout_steps)
    B = batch size (i.e. config.num_envs)
    N = num agents (i.e. config.num_agents)

    Arguments
    ---------
    batch:
        The batch to split and pad. Shape=(T, B, N, ...).
    seq_idxs:
        The indices of each sequence chunk for each batch dim. The number of chunks for
        each batch dim may vary (i.e. due to episodes terminated at different steps,
        etc.) Shape=(B, num_seqs_chunks along batch dim).
    max_seq_len:
        The maximum sequence length to pad to.
    padding_value:
        The value to use for padding.

    Returns
    -------
    batch:
        The batch split into sequences and padded to `max_seq_len`.
        Shape=(`max_seq_len`, total number of sequence chunks, N, ...).

    """
    sequences = []
    for batch_idx, chunk_idxs in enumerate(seq_idxs):
        for chunk_idx in chunk_idxs:
            sequences.append(batch[chunk_idx, batch_idx])

    b_seq = nn.utils.rnn.pad_sequence(
        sequences, batch_first=False, padding_value=padding_value
    )
    if b_seq.shape[0] < max_seq_len:
        padding = [0, 0] * len(b_seq.shape)
        padding[-1] = max_seq_len - b_seq.shape[0]
        b_seq = F.pad(b_seq, padding, mode="constant", value=0.0)

    return b_seq


def split_lstm_state_batch(
    lstm_state_batch: torch.tensor,
    seq_idxs: List[List[torch.tensor]],
) -> torch.tensor:
    """Split batch of LSTM states according to sequences.

    The batch is first split into episodes, then each episode is split into sequences
    of length `max_seq_len`. The LSTM state at the start of each sequence is then
    extracted from the batch.

    T = batch steps (i.e. config.num_rollout_steps)
    B = batch size (i.e. config.num_envs)
    N = num agents (i.e. config.num_agents)

    Arguments
    ---------
    lstm_state_batch:
        The batch to split and pad. Shape=(T, num_lstm_layers, B, N, lstm_size).
    seq_idxs:
        The indices of each sequence chunk for each batch dim. The number of chunks for
        each batch dim may vary (i.e. due to episodes terminated at different steps,
        etc.) Shape=(B, num_seqs_chunks along batch dim).

    Returns
    -------
    init_lstm_states:
        The initial lstm state for each sequence.
        Shape=(num_lstm_layers, total number of sequence chunks, N, lstm_size).

    """
    init_lstm_states = []
    for batch_idx, chunk_idxs in enumerate(seq_idxs):
        for chunk_idx in chunk_idxs:
            init_lstm_states.append(lstm_state_batch[chunk_idx[0], :, batch_idx])

    init_lstm_states = torch.stack(init_lstm_states, dim=1)
    return init_lstm_states
