# POSGGym Baselines
Implementations of planning and reinforcement learning algorithms for [POSGGym](https://github.com/RDLLab/posggym) environments.

## Installation

Install by cloning repo and installing locally with pip.

``` bash
git clone git@github.com:RDLLab/posggym-baselines.git
cd posggym-baselines
pip install -e .
```

To install all requirements used for the baseline exps in the `baseline_exps` run:

```bash
pip install -e .[exps]
```

Note, you will also need [jupyter notebook](https://jupyter.org/install) installed to view and run analysis notebooks.

## Planning Algorithms

POSGGym-Baselines includes implementations of a number partially observable, multi-agent planning algorithms, including:

| Algorithm | Action Space | Observation Space |
| --------- | ------------ | ----------------- |
| POMCP     | Discrete     | Discrete          |
| I-NTMCP   | Discrete     | Discrete          |
| I-POMCP   | Discrete     | Discrete          |
| POTMMCP   | Discrete     | Discrete          |


Each planner requires slightly different inputs for initialization, but once loaded each can be used in the same manner using the `step` and `reset` functions. For a full example see the code for the POMCP algorithm below.


### POMCP

POMCP is an algorithm original designed for planning in single-agent POMDPs, the implementation in this library extends it to the multi-agent setting by treating the other agent as random noise during planning.

The following is an example of using POMCP in the `Driving-v1` environment with the other agent in the environment being random.

```python
import posggym
import posggym.agents as pga
from posggym_baselines.planning import (
    POMCP,
    MCTSConfig,
    RandomSearchPolicy,
    load_posggym_agents_search_policy,
)

env = posggym.make("Driving-v1")

config = MCTSConfig(
    discount=0.95,           # expected return discount factor
    search_time_limit=0.1,   # per step search time
    c=1.414,                 # ~ math.sqrt(2)
    truncated=False,         # use monte-carlo rollouts
    action_selection="ucb",  # search action selection policy
)

planning_agent_id = env.possible_agents[0]
other_agent_id = env.possible_agents[1]

planner = POMCP(
    env.model,
    planning_agent_id,
    config,
    search_policy=RandomSearchPolicy(env.model, planning_agent_id),
)

obs, infos = env.reset()
all_done = False

planner.reset()
while not all_done:
    actions = {
        planning_agent_id: planner.step(obs[planning_agent_id]),
        other_agent_id: env.action_spaces[other_agent_id].sample()
    }
    obs, rewards, terms, truncs, all_done, infos = env.step(actions)

env.close()
planner.close()
```

*Reference*:

- [Silver, D. & Veness, J. Monte-Carlo Planning in Large POMDPs. Advances in Neural Information Processing Systems 23, 2164–2172 (2010)](https://proceedings.neurips.cc/paper_files/paper/2010/hash/edfbe1afcf9246bb0d40eb4d8027d90f-Abstract.html)


### INTMCP

INTMCP is an algorithm for planning in two-agent environments. It models the problem as an I-POMDP and constructs the policy for the other agent online.

**Note** I-NTMCP currently only works for environments with two-agents. It is technically feasible to extend it to work with more than two-agents but this would significantly increase the computational cost and implementation complexity.

The following is an example of initializing INTMCP in the `Driving-v1` environment. Once initialized it can be used as per the POMCP example above.

```python
import posggym
import posggym.agents as pga
from posggym_baselines.planning import INTMCP, MCTSConfig

env = posggym.make("Driving-v1")

config = MCTSConfig(
    discount=0.95,           # expected return discount factor
    search_time_limit=0.1,   # per step search time
    c=1.414,                 # ~ math.sqrt(2)
    truncated=False,         # use monte-carlo rollouts
    action_selection="ucb",  # search action selection policy
)

planning_agent_id = env.possible_agents[0]

planner = INTMCP.initialize(
    env.model,
    planning_agent_id,
    config,
    nesting_level=1,
    search_policies=None,       # Use RandomSearchPolicy
)
```

*Reference*:

- [Schwartz, J., Zhou, R. & Kurniawati, H. Online planning for interactive-pomdps using nested monte carlo tree search. in International Conference on Intelligent Robots and Systems 8770–8777 (IEEE, 2022).](https://ieeexplore.ieee.org/abstract/document/9981713?casa_token=30NsbqmPOrIAAAAA:WIvn8YsrqIFDPJxBSjPXD6ethl8XttnYPs7rw9Yv8GSSFcYshxnj_UKQZ0-L549A2YBjJtpjS5g) | [Code](https://github.com/RDLLab/i-ntmcp)


### IPOMCP

IPOMCP is an algorithm originally designed for planning in *Open and Typed* multi-agent environments with many agents. In this setting agents may enter and leave the environment at any point during an episode. Our implementation is a simplified version of IPOMCP for environments with few agents and without explicit modelling of agents coming and going.

Our implementation if closer to the CI-POMCP algorithm which adapted IPOMCP to environments with fewer agents but with explicit communication.

The following is an example of initializing IPOMCP in the `Driving-v1` environment assuming the other agent is using one of the shorted path policies. Once initialized it can be used as per the POMCP example above.

```python
import posggym
import posggym.agents as pga
from posggym_baselines.planning import (
    IPOMCP,
    MCTSConfig,
    RandomSearchPolicy,
    OtherAgentMixturePolicy
)

env = posggym.make("Driving-v1")

config = MCTSConfig(
    discount=0.95,           # expected return discount factor
    search_time_limit=0.1,   # per step search time
    c=1.414,                 # ~ math.sqrt(2)
    truncated=False,         # use monte-carlo rollouts
    action_selection="ucb",  # search action selection policy
)

planning_agent_id = env.possible_agents[0]
other_agent_id = env.possible_agents[1]

planner_other_agent_policies = OtherAgentMixturePolicy.load_posggym_agents_policy(
    env.model,
    other_agent_id,
    [
        "Driving-v1/A0Shortestpath-v0",
        "Driving-v1/A40Shortestpath-v0",
        "Driving-v1/A60Shortestpath-v0",
        "Driving-v1/A80Shortestpath-v0",
        "Driving-v1/A100Shortestpath-v0",
    ]
)

planner = IPOMCP(
    env.model,
    planning_agent_id,
    config,
    other_agent_policies={other_agent_id: planner_other_agent_policies},
    search_policies=RandomSearchPolicy(env.model, planning_agent_id)
)
```

*Reference*:

- **Original IPOMCP Paper** [Eck, A., Shah, M., Doshi, P. & Soh, L.-K. Scalable decision-theoretic planning in open and typed multiagent systems. in AAAI Conference on Artificial Intelligence vol. 34 7127–7134 (2020)](https://ojs.aaai.org/index.php/AAAI/article/view/6200) | [Code](https://github.com/OberlinAI/ScalableOASYS)
- **CI-POMCP Paper** [Kakarlapudi, A., Anil, G., Eck, A., Doshi, P. & Soh, L.-K. Decision-Theoretic Planning with Communication in Open Multiagent Systems. in Uncertainty in Artificial Intelligence 938–948 (PMLR, 2022).
](https://proceedings.mlr.press/v180/kakarlapudi22a.html) | [Code](https://github.com/OberlinAI/CommunicativeOASYS)


### POTMMCP

POTMMCP is an algorithm designed for planning in *Typed* multi-agent environments. Its key distinguishing feature is the use of a meta-policy for guiding search. The meta-policy is used to select which of the types should be used to guide the search from each belief.

The following is an example of initializing POMCP in the `Driving-v1` environment assuming the other agent is using one of the shorted path policies. In this example we define the meta-policy to select the same policy as the other agent is believed to be using.

Once initialized it can be used as per the POMCP example above.

```python
import posggym
import posggym.agents as pga
from posggym_baselines.planning import (
    POTMMCP,
    MCTSConfig,
    POTMMCPMetaPolicy,
    OtherAgentMixturePolicy
)

env = posggym.make("Driving-v1")

config = MCTSConfig(
    discount=0.95,            # expected return discount factor
    search_time_limit=0.1,    # per step search time
    c=1.25,
    truncated=True,           # use value function of search policy, if available
    action_selection="pucb",  # search action selection policy
)

planning_agent_id = env.possible_agents[0]
other_agent_id = env.possible_agents[1]

# set of possible types/policies for the other agent
other_agent_policy_ids = [
    "Driving-v1/A0Shortestpath-v0",
    "Driving-v1/A40Shortestpath-v0",
    "Driving-v1/A60Shortestpath-v0",
    "Driving-v1/A80Shortestpath-v0",
    "Driving-v1/A100Shortestpath-v0",
]

# define the meta-policy, mapping policy ID of other agent to the ID of the
# policy to use for guiding planning
meta_policy = {pi_id: {pi_id: 1.0} for pi_id in other_agent_policy_ids}

search_policy = POTMMCPMetaPolicy.load_posggym_agents_meta_policy(
    env.model, planning_agent_id, meta_policy
)

planner_other_agent_policies = OtherAgentMixturePolicy.load_posggym_agents_policy(
    env.model, other_agent_id, other_agent_policy_ids
)

planner = IPOMCP(
    env.model,
    planning_agent_id,
    config,
    other_agent_policies={other_agent_id: planner_other_agent_policies},
    search_policies=search_policy
)
```

For more principled methods for generating the meta-policy please refer to the POTMMCP paper, and the `baseline_exp/potmmcp_meta_policy.ipynb` notebook.

*Reference*:

- [Schwartz, J., Kurniawati, H. & Hutter, M. Combining a Meta-Policy and Monte-Carlo Planning for Scalable Type-Based Reasoning in Partially Observable Environments. arXiv preprint arXiv:2306.06067 (2023).
](https://arxiv.org/abs/2306.06067) | [Code](https://github.com/Jjschwartz/potmmcp)


## Reinforcement Learning Algorithms

POSGGym-Baselines includes an implementation of a number of different multi-agent RL algorithms. Every algorithm is used to train one or more policies with PPO used to train each individual policy.

Implemented algorithms include:

- **Self-Play (SP)** - trains a population of policies where each policy is trained using self-play (i.e. against itself only).
- **Self-Play plus Best-Response (SP-BR)** - Same as SP but includes training a Best-Response (BR) against the population of policies. The BR is a single policy which is trained against each policy in the SP population.
- **K-Level Reasoning (KLR)** - trains a population of K-Level Reasoning policies where policies are trained in a hierarchy. The level 0 policy is trained against the uniform random policy, while for level l>0, the level l policy is trained against the level l-1 policy. All policies are trained synchronously.
- **K-Level Reasoning plus Best-Response (KLR-BR)** - Same as KLR but includes training a Best-Response (BR) against the population of KLR policies.

An example implementation is provided in the `baseline_exps/train_population_policies.py` file, which can be used to train a population in a few of the POSGGym environments:

```bash
cd posggym-baselines
# see options
python baseline_exps/train_population_policies.py --help
# train SP population with 4 policies in Driving-v1 environment
python baseline_exps/train_population_policies.py \
    SP \
    --full_env_id Driving-v1 \
    --pop_size 4
```

Note that the script includes support for logging via Tensorboard and WandB. It also supports using multiple CPUs for training and LSTM policies.


## Citation

You can cite POSGGym as:

```bibtex
@misc{schwartzPOSGGym2023,
    title = {POSGGym},
    urldate = {2023-08-08},
    author = {Schwartz, Jonathon and Newbury, Rhys and Kurniawati, Hanna},
    year = {2023},
}
```
