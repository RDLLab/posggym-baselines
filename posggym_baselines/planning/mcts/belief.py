import random
from typing import List, Optional, Dict, Callable, TypeVar, Generic

import posggym.model as M
from posggym.utils.history import JointHistory
from posggym.agents.policy import PolicyState


ParticleType = TypeVar("ParticleType")


class HistoryPolicyState:
    """A History Policy state.

    Consists of:
    1. the environment state
    2. the joint history of each agent up to given state
    3. the state of the other agents policies

    The state of the other agent policies is represented as a tuple with a PolicyState
    (i.e. a dictionary) for each other agent. This representation allows for arbitrary
    policy state representations for each agent, including representing multiple other
    agent policies in a single policy, and/or policy network states if the other
    agent's policy is a neural network.

    """

    def __init__(
        self,
        state: M.StateType,
        history: JointHistory,
        policy_state: Dict[str, PolicyState],
    ):
        self.state = state
        self.history = history
        self.policy_state = policy_state
        self.t = len(history)

    def __str__(self):
        return f"[s={self.state}, h={self.history}, pi={self.policy_state}]"

    def __repr__(self):
        return self.__str__()


class ParticleBelief(Generic[ParticleType]):
    """A belief represented by state particles."""

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng if rng is not None else random.Random()
        self.particles = []

    def sample(self) -> ParticleType:
        return self.rng.choice(self.particles)

    def add_particle(self, state: ParticleType):
        self.particles.append(state)

    def size(self) -> int:
        return len(self.particles)

    def clear(self) -> None:
        self.particles.clear()


class BeliefRejectionSampler:
    """Reinvigorates a belief using rejection sampling.

    Reinvigorate function takes additional optional and requires kwargs:

    'use_rejected_samples': bool, Optional
        whether to use sampling without rejection to ensure 'num_particles'
        additional particles are added to the belief in the case that
        'num_particles' valid particles aren't sampled within 'sample_limit'
        samples using rejection sampling (default=False)
    """

    def __init__(self, model: M.POSGModel, sample_limit: int = 1000):
        self._model = model
        self._sample_limit = sample_limit

    def reinvigorate(
        self,
        agent_id: str,
        belief: ParticleBelief,
        action: M.ActType,
        obs: M.ObsType,
        num_particles: int,
        parent_belief: ParticleBelief,
        joint_action_fn: Callable,
        joint_update_fn: Callable,
        **kwargs,
    ):
        """Reinvigorate belief given action performed and observation recieved.

        In general this involves adding additional particles to the belief that
        are consistent with the action and observation.

        Arguments:
        ---------
        agent_id
            ID of the agent to reinvigorate belief of
        belief
            The belief to reinvigorate given last action and observation
        action
            Action performed by agent
        obs
            The observation recieved by agent
        num_particles
            the number of additional particles to sample
        parent_belief
            the parent belief of the belief being reinvigorated
        joint_action_fn : Callable[
                [HistoryPolicyState, M.ActType], M.JointAction
            ]
            joint action selection function
        joint_update_fn : Callable[
            [HistoryPolicyState, M.JointAction, M.JointObservation],
            potmmcp.policy.PolicyHiddenStates
        ]
            update function for policies

        """
        new_particles = self._rejection_sample(
            agent_id,
            action,
            obs,
            parent_belief=parent_belief,
            num_samples=num_particles,
            joint_action_fn=joint_action_fn,
            joint_update_fn=joint_update_fn,
            use_rejected_samples=kwargs.get("use_rejected_samples", False),
        )
        for p in new_particles:
            belief.add_particle(p)

    def _rejection_sample(
        self,
        agent_id: str,
        action: M.ActType,
        obs: M.ObsType,
        parent_belief: ParticleBelief,
        num_samples: int,
        joint_action_fn: Callable,
        joint_update_fn: Callable,
        use_rejected_samples: bool,
    ) -> List[HistoryPolicyState]:
        sample_count = 0
        retry_count = 0
        rejected_samples = []
        samples = []

        while sample_count < num_samples and retry_count < max(
            num_samples, self._sample_limit
        ):
            hp_state = parent_belief.sample()
            joint_action = joint_action_fn(hp_state, action)
            joint_step = self._model.step(hp_state.state, joint_action)
            joint_obs = joint_step.observations

            if joint_obs[agent_id] != obs and not use_rejected_samples:
                retry_count += 1
                continue

            new_history = hp_state.history.extend(joint_action, joint_obs)
            next_policy_state = joint_update_fn(hp_state, joint_action, joint_obs)
            next_hp_state = HistoryPolicyState(
                joint_step.state,
                new_history,
                next_policy_state,
            )

            if joint_obs[agent_id] == obs:
                samples.append(next_hp_state)
                sample_count += 1
            else:
                if use_rejected_samples:
                    rejected_samples.append(next_hp_state)
                retry_count += 1

        if sample_count < num_samples and use_rejected_samples:
            num_missing = num_samples - sample_count
            samples.extend(rejected_samples[:num_missing])

        return samples
