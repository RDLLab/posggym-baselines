"""The core PPO learner."""
import os
import time
from datetime import timedelta
from multiprocessing.queues import Empty, Full
from typing import TYPE_CHECKING, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

import posggym_baselines.ppo.utils as ppo_utils
from posggym_baselines.ppo.eval import run_eval_worker
from posggym_baselines.ppo.worker import run_rollout_worker
from posggym_baselines.utils import logger


if TYPE_CHECKING:
    from posggym_baselines.ppo.config import PPOConfig


class PPOLearner:
    """Distributed PPO learner."""

    def __init__(self, config: "PPOConfig"):
        self.config = config

        if config.disable_logging:
            self.writer = logger.NullLogger(config)
        else:
            self.writer = logger.TensorBoardLogger(config)

        self.policies = config.load_policies(device=config.device)
        for policy in self.policies.values():
            policy.share_memory()

        self.optimizers = {
            policy_id: optim.Adam(
                policy.parameters(), lr=config.learning_rate, eps=1e-5
            )
            for policy_id, policy in self.policies.items()
        }

    def train(
        self,
        worker_recv_queues: List[mp.Queue],
        worker_send_queues: List[mp.Queue],
        eval_recv_queue: Optional[mp.Queue],
        eval_send_queue: Optional[mp.Queue],
        termination_event: mp.Event,
    ):
        """Run PPO training.

        Arguments
        ---------
        worker_recv_queues:
            A list of queues for sending data to each worker.
        worker_send_queues:
            A list of queues for receiving data from each worker.
        eval_recv_queues:
            Queue for sending data to eval worker.
        eval_send_queues:
            Queue for receiving data from eval worker.
        termination_event:
            An event for signalling termination of training
        """
        assert len(worker_recv_queues) == self.config.num_workers
        assert len(worker_send_queues) == self.config.num_workers
        if self.config.eval_interval > 0:
            assert eval_recv_queue is not None
            assert eval_send_queue is not None

        global_step = 0
        update = 1
        train_start_time = time.time()
        sps_start_time = train_start_time
        while update < self.config.num_updates + 1 and not termination_event.is_set():
            if self.config.anneal_lr:
                frac = 1.0 - (update - 1.0) / self.config.num_updates
                lrnow = frac * self.config.learning_rate
                for optimizer in self.optimizers.values():
                    optimizer.param_groups[0]["lr"] = lrnow

            # current model state to share with workers
            policy_states = {
                policy_id: policy.state_dict()
                for policy_id, policy in self.policies.items()
            }

            experience_collection_start_time = time.time()
            # signal workers to collect next batch of experience
            for i in range(self.config.num_workers):
                worker_recv_queues[i].put(policy_states)

            # wait for workers to finish collecting experience
            worker_batches = []
            next_worker_id = 0
            while (
                next_worker_id < self.config.num_workers
                and not termination_event.is_set()
            ):
                try:
                    # wait for worker batch, with timeout for checking for termination
                    batch = worker_send_queues[next_worker_id].get(timeout=1)
                    worker_batches.append(batch)
                    next_worker_id += 1
                except Empty:
                    pass
            if termination_event.is_set():
                break

            # combine worker batches into single batch
            combined_batch = ppo_utils.combine_batches(worker_batches, self.config)
            # free any references to worker tensors stored in shared memory
            worker_batches = []

            experience_collection_time = time.time() - experience_collection_start_time
            global_step += self.config.batch_size

            # update policies
            learning_start_time = time.time()
            self.update(combined_batch, global_step)
            learning_time = time.time() - learning_start_time

            # record timing and other general stats
            sps = int(global_step / (time.time() - sps_start_time))
            print(f"learner: Update: {update}/{self.config.num_updates} SPS: {sps}")

            self.writer.log_scalar("charts/update", update, global_step)
            self.writer.log_scalar(
                "charts/learning_rate",
                list(self.optimizers.values())[0].param_groups[0]["lr"],
                global_step,
            )
            self.writer.log_scalar("charts/SPS", sps, global_step)
            self.writer.log_scalar(
                "charts/collection_time", experience_collection_time, global_step
            )
            self.writer.log_scalar("charts/learning_time", learning_time, global_step)

            # log policy episode stats (logging in policy ID order)
            policy_returns = {}
            for policy_id in self.config.get_all_policy_ids():
                if policy_id not in combined_batch.get("policy_stats", {}):
                    continue
                policy_stats = combined_batch["policy_stats"][policy_id]
                prefix = f"policy_stats/{policy_id}"
                for stat_name, stat_value in policy_stats.items():
                    self.writer.log_scalar(
                        f"{prefix}/{stat_name}", stat_value, global_step
                    )
                    if stat_name == "mean_episode_return":
                        policy_returns[policy_id] = stat_value

            if policy_returns:
                # display progress in stdout
                training_time = timedelta(seconds=int(time.time() - train_start_time))
                output = [f"learner: time={training_time} global_step={global_step}"]
                output += [
                    f"  {policy_id}: {policy_return:.2f}"
                    for policy_id, policy_return in policy_returns.items()
                ]
                print("\n".join(output))

            if (
                not self.config.disable_logging
                and self.config.save_interval > 0
                and (
                    update % self.config.save_interval == 0
                    or update == self.config.num_updates
                )
            ):
                print("learner: Saving policy models")
                self.save(global_step, update)

            if self.config.eval_interval > 0 and (
                update % self.config.eval_interval == 0
                or update in (1, self.config.num_updates)
            ):
                print("learner: Evaluating policies")
                eval_time = time.time()
                self.evaluate(
                    global_step,
                    eval_recv_queue,
                    eval_send_queue,
                    termination_event,
                    update == self.config.num_updates,
                )
                # don't count eval time towards SPS
                sps_start_time += time.time() - eval_time

            video_upload_time = time.time()
            self.writer.upload_videos(global_step)
            sps_start_time += time.time() - video_upload_time

            update += 1

    def update(self, batch: Dict[str, torch.tensor], global_step: int):
        """Update the policies using the batch of experience."""
        # calculate advantages and monte-carlo returns
        rewards_buf, dones_buf, values_buf = (
            batch["rewards"],
            batch["dones"],
            batch["values"],
        )
        advantages = torch.zeros_like(rewards_buf)
        lastgaelam = 0
        for t in reversed(range(self.config.seq_len)):
            nextnonterminal = 1.0 - dones_buf[t + 1]
            nextvalues = values_buf[t + 1]
            delta = (
                rewards_buf[t]
                + self.config.gamma * nextvalues * nextnonterminal
                - values_buf[t]
            )
            advantages[t] = lastgaelam = delta + (
                self.config.gamma
                * self.config.gae_lambda
                * nextnonterminal
                * lastgaelam
            )
        returns = advantages + values_buf[:-1]
        if self.config.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch["advantages"] = advantages
        batch["returns"] = returns

        # remove final additional timestep from batch "dones" and "values"
        batch["dones"] = batch["dones"][:-1]
        batch["values"] = batch["values"][:-1]

        # split batch by policy
        policy_batches = ppo_utils.split_batch_by_policy(batch, self.config)

        # filter out experience for non-training policies
        if self.config.filter_experience:
            policy_batches = ppo_utils.filter_policy_batches_by_partner_dist(
                policy_batches, self.config
            )

        # update each policy
        for policy_id, policy in self.policies.items():
            if policy_id not in policy_batches:
                # no experience collected for this policy
                continue
            policy_batch = policy_batches[policy_id]
            batch_seq_len, num_seqs_in_batch = policy_batch["obs"].shape[:2]
            batch_size = num_seqs_in_batch * batch_seq_len

            # flatten the batch
            b_obs = (
                policy_batch["obs"]
                .reshape((-1,) + self.config.obs_space.shape)
                .to(self.config.device)
            )
            b_logprobs = policy_batch["logprobs"].reshape(-1).to(self.config.device)
            b_actions = (
                policy_batch["actions"]
                .reshape((-1,) + self.config.act_space.shape)
                .to(self.config.device)
            )
            b_advantages = policy_batch["advantages"].reshape(-1).to(self.config.device)
            b_returns = policy_batch["returns"].reshape(-1).to(self.config.device)
            b_dones = policy_batch["dones"].reshape(-1).to(self.config.device)
            b_values = policy_batch["values"].reshape(-1).to(self.config.device)
            b_initial_states = policy_batch["initial_lstm_states"]
            b_initial_states = (
                b_initial_states[0].to(self.config.device),
                b_initial_states[1].to(self.config.device),
            )

            seq_indxs = np.arange(num_seqs_in_batch)
            flat_indxs = np.arange(batch_size).reshape(batch_seq_len, num_seqs_in_batch)

            clipfracs = []
            approx_kl, old_approx_kl, unclipped_grad_norm = 0, 0, 0
            entropy_loss, pg_loss, v_loss, loss = 0, 0, 0, 0
            for epoch in range(self.config.update_epochs):
                np.random.shuffle(seq_indxs)

                # minibatch update, using data from randomized subset of sequences
                for start in range(
                    0, num_seqs_in_batch, self.config.minibatch_num_seqs
                ):
                    end = start + self.config.minibatch_num_seqs
                    mb_seq_indxs = seq_indxs[start:end]
                    # be really careful about the index
                    mb_indxs = flat_indxs[:, mb_seq_indxs].ravel()

                    _, newlogprob, entropy, newvalue, _ = policy.get_action_and_value(
                        b_obs[mb_indxs],
                        (
                            b_initial_states[0][:, mb_seq_indxs],
                            b_initial_states[1][:, mb_seq_indxs],
                        ),
                        b_dones[mb_indxs],
                        b_actions.long()[mb_indxs],
                    )
                    logratio = newlogprob - b_logprobs[mb_indxs]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.config.clip_coef)
                            .float()
                            .mean()
                            .item()
                        ]

                    # Policy loss
                    pg_loss1 = -b_advantages[mb_indxs] * ratio
                    pg_loss2 = -b_advantages[mb_indxs] * torch.clamp(
                        ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.config.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_indxs]) ** 2
                        v_clipped = b_values[mb_indxs] + torch.clamp(
                            newvalue - b_values[mb_indxs],
                            -self.config.clip_coef,
                            self.config.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_indxs]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_indxs]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss
                        - self.config.ent_coef * entropy_loss
                        + self.config.vf_coef * v_loss
                    )

                    self.optimizers[policy_id].zero_grad()
                    loss.backward()
                    unclipped_grad_norm = nn.utils.clip_grad_norm_(
                        policy.parameters(), self.config.max_grad_norm
                    )
                    self.optimizers[policy_id].step()

                if (
                    self.config.target_kl is not None
                    and approx_kl > self.config.target_kl
                ):
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            # record learning statistics
            prefix = f"losses/{policy_id}"
            self.writer.log_scalar(f"{prefix}/loss", loss.item(), global_step)
            self.writer.log_scalar(f"{prefix}/value_loss", v_loss.item(), global_step)
            self.writer.log_scalar(f"{prefix}/policy_loss", pg_loss.item(), global_step)
            self.writer.log_scalar(
                f"{prefix}/entropy", entropy_loss.item(), global_step
            )
            self.writer.log_scalar(
                f"{prefix}/old_approx_kl", old_approx_kl.item(), global_step
            )
            self.writer.log_scalar(f"{prefix}/approx_kl", approx_kl.item(), global_step)
            self.writer.log_scalar(
                f"{prefix}/clipfrac", np.mean(clipfracs), global_step
            )
            self.writer.log_scalar(
                f"{prefix}/explained_variance", explained_var, global_step
            )
            self.writer.log_scalar(
                f"{prefix}/unclipped_grad_norm", unclipped_grad_norm.item(), global_step
            )
            self.writer.log_scalar(
                f"{prefix}/num_sequences", num_seqs_in_batch, global_step
            )

    def evaluate(
        self,
        global_step: int,
        eval_recv_queue: mp.Queue,
        eval_send_queue: mp.Queue,
        termination_event: mp.Event,
        final_eval: bool,
    ):
        """Run evaluation of policies."""
        policy_states_clone = {
            policy_id: {
                k: v.clone().to(self.config.eval_device).share_memory_()
                for k, v in policy.state_dict().items()
            }
            for policy_id, policy in self.policies.items()
        }
        # queue up next eval, queue may be full if enough of previous evals
        # havn't finished, in which case wait for one to finish (to avoid
        # excessive memory usage, default maxsize=100)
        reported_wait = False
        print(f"learner: Starting eval for global_step={global_step} {final_eval=}")
        while not termination_event.is_set():
            try:
                eval_recv_queue.put((policy_states_clone, global_step), timeout=1)
                break
            except Full:
                if not reported_wait:
                    print("learner: Eval queue full, waiting for eval to finish")
                    reported_wait = True
                pass

        # check if previous eval finished and log results
        evals_to_log = True
        while not termination_event.is_set() and evals_to_log:
            try:
                eval_results = eval_send_queue.get(timeout=1)
                print(
                    f"learner: Eval finished for "
                    f"global_step={eval_results['global_step']} "
                    f"eval_time={eval_results['eval_time']:.2f} seconds"
                )

                self.writer.log_scalar(
                    "charts/eval_time",
                    eval_results["eval_time"],
                    eval_results["global_step"],
                )
                for k, v in eval_results["metrics"].items():
                    if isinstance(v, np.ndarray):
                        self.writer.log_matrix(
                            f"eval/{k}", v, eval_results["global_step"]
                        )
                    elif isinstance(v, plt.Figure):
                        self.writer.log_figure(
                            f"eval/{k}", v, eval_results["global_step"]
                        )
                    else:
                        self.writer.log_scalar(
                            f"eval/{k}", v, eval_results["global_step"]
                        )

                evals_to_log = not eval_send_queue.empty()
                if final_eval:
                    evals_to_log = eval_results["global_step"] != global_step

            except Empty:
                evals_to_log = final_eval

    def save(self, global_step: int, update: int):
        for policy_id, policy in self.policies.items():
            torch.save(
                {
                    "model": policy.state_dict(),
                    "optimizer": self.optimizers[policy_id].state_dict(),
                    "global_step": global_step,
                    "update": update,
                    "config": self.config.asdict(),
                },
                os.path.join(
                    self.config.log_dir, f"checkpoint_{update}_{policy_id}.pt"
                ),
            )

    def close(self):
        self.writer.close()


def run_learner(
    config: "PPOConfig",
    worker_recv_queues: List[mp.Queue],
    worker_send_queues: List[mp.Queue],
    eval_recv_queue: Optional[mp.Queue],
    eval_send_queue: Optional[mp.Queue],
    termination_event: mp.Event,
):
    """Run PPO learner process.

    Arguments
    ---------
    config:
        The configuration information.
    worker_recv_queues:
        A list of queues for sending data to each worker.
    worker_send_queues:
        A list of queues for receiving data from each worker.
    eval_recv_queues:
        Queue for sending data to eval worker.
    eval_send_queues:
        Queue for receiving data from eval worker.
    termination_event:
        An event for signalling termination of training
    """
    print(f"learner: Starting learner process, Device={config.device}")

    torch.set_num_threads(1)

    ppo_learner = PPOLearner(config)

    print("learner: Starting training loop...")
    ppo_learner.train(
        worker_recv_queues,
        worker_send_queues,
        eval_recv_queue,
        eval_send_queue,
        termination_event,
    )
    ppo_learner.close()

    if termination_event.is_set():
        print("learner: Training terminated early due to error in another process.")
    else:
        # set termination event so workers know to stop
        # and to signal to main that learner finished as expected
        termination_event.set()
        print("learner: Training complete.")

    print("learner: All done")


def run_ppo(config: "PPOConfig"):
    print("Running PPO:")
    print(f"Env-id: {config.env_id}")
    print(f"Observation space: {config.obs_space}")
    print(f"Action space: {config.act_space}")

    # Spawn workers
    # `fork` not supported by CUDA
    # https://pytorch.org/docs/main/notes/multiprocessing.html#cuda-in-multiprocessing
    # must use context to set start method
    mp_ctxt = mp.get_context("spawn")

    # global termination event
    termination_event = mp_ctxt.Event()

    # create rollout workers and queues for communication
    worker_recv_queues = []
    worker_send_queues = []
    workers = []
    for worker_id in range(config.num_workers):
        worker_recv_queues.append(mp_ctxt.Queue())
        worker_send_queues.append(mp_ctxt.Queue())
        worker = mp_ctxt.Process(
            target=run_rollout_worker,
            args=(
                worker_id,
                config,
                worker_recv_queues[worker_id],
                worker_send_queues[worker_id],
                termination_event,
            ),
        )
        worker.start()
        workers.append(worker)

    # create eval worker and queues for communication
    if config.eval_interval > 0:
        # set max size so learner can't get too far ahead of eval
        eval_recv_queue = mp_ctxt.Queue(maxsize=5)
        eval_send_queue = mp_ctxt.Queue()
        eval_worker = mp_ctxt.Process(
            target=run_eval_worker,
            args=(
                config,
                eval_recv_queue,
                eval_send_queue,
                termination_event,
            ),
        )
        eval_worker.start()
    else:
        eval_recv_queue = None
        eval_send_queue = None
        eval_worker = None

    # run learner
    learner = mp_ctxt.Process(
        target=run_learner,
        args=(
            config,
            worker_recv_queues,
            worker_send_queues,
            eval_recv_queue,
            eval_send_queue,
            termination_event,
        ),
    )
    learner.start()

    print("main: Waiting for worker and learner processes to finish.")
    worker_crashed = False
    while not worker_crashed:
        time.sleep(1)
        if not learner.is_alive():
            if termination_event.is_set():
                print("main: Learner process finished training.")
            else:
                print("main: Learner process crashed.")
            break
        for i, worker in enumerate(workers):
            if not worker.is_alive():
                print(f"main: Worker {i} process crashed.")
                worker_crashed = True
        if eval_worker is not None and not eval_worker.is_alive():
            print("main: Eval worker process crashed.")
            worker_crashed = True

    print("main: Training finished, shutting down.")
    if not termination_event.is_set():
        termination_event.set()

    print("main: Draining worker queues.")
    for q in worker_recv_queues + worker_send_queues:
        while not q.empty():
            q.get()

    print("main: Draining eval queue.")
    for q in (eval_recv_queue, eval_send_queue):
        while q is not None and not q.empty():
            q.get()

    print("main: Joining workers.")
    for i in range(config.num_workers):
        workers[i].join()

    print("main: Joining eval worker.")
    if eval_worker is not None:
        eval_worker.join()

    print("main: Joining learner.")
    learner.join()

    print("main: Worker and learner processes successfully joined.")
    print("main: Cleaning up communication queues.")
    for q in worker_recv_queues + worker_send_queues:
        q.close()
    if eval_recv_queue is not None:
        eval_recv_queue.close()
    if eval_send_queue is not None:
        eval_send_queue.close()

    print("main: All done")
