"""A node in the search tree."""
from typing import Dict, List, Optional

import posggym.model as M
from posggym.agents.policy import PolicyState

from posggym_baselines.planning.belief import ParticleBelief


class Node:
    """A node in the search tree."""

    # class variable
    node_count = 0

    def __init__(self):
        self.nid = Node.node_count
        Node.node_count += 1

    def __hash__(self):
        return self.nid

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.nid == other.nid


class ObsNode(Node):
    """An observation node in the search tree."""

    def __init__(
        self,
        parent: Optional["ActionNode"],
        obs: M.ObsType,
        t: int,
        belief: ParticleBelief,
        action_probs: Dict[M.ActType, float],
        search_policy_state: Optional[PolicyState] = None,
        init_value: float = 0.0,
        init_visits: int = 0,
        is_absorbing: bool = False,
    ):
        super().__init__()
        self.parent: "ActionNode" = NullNode() if parent is None else parent
        self.obs = obs
        self.t = t
        self.belief = belief
        self.action_probs = action_probs
        self.search_policy_state = {}
        if search_policy_state is not None:
            self.search_policy_state = search_policy_state
        self.value = init_value
        self.visits = init_visits
        self.is_absorbing = is_absorbing
        self.children: List["ActionNode"] = []

    def get_child(self, action: M.ActType) -> "ActionNode":
        """Get child node for given action value."""
        for action_node in self.children:
            if action_node.action == action:
                return action_node
        raise AssertionError(f"ObsNode {str(self)} has no child node for {action=}")

    def has_child(self, action: M.ActType) -> bool:
        """Check if this obs node has a child node matching action."""
        return any(action_node.action == action for action_node in self.children)

    def add_child(self, action: M.ActType) -> "ActionNode":
        """Add a child node for given action."""
        action_node = ActionNode(
            parent=self,
            action=action,
            prob=self.action_probs.get(action, 0.0),
            t=self.t,
            init_value=0.0,
            init_visits=0,
            init_total_value=0.0,
        )
        self.children.append(action_node)
        return action_node

    def policy_str(self) -> str:
        """Get policy in nice str format."""
        action_probs = [f"{a}: {prob:.2f}" for a, prob in self.action_probs.items()]
        return "{" + ",".join(action_probs) + "}"

    def clear_belief(self):
        """Delete all particles in belief of node."""
        self.belief.clear()

    def is_root(self) -> bool:
        """Return true if this node is a root node."""
        return isinstance(self.parent, NullNode)

    def __str__(self):
        return (
            f"N{self.nid}"
            f"\no={self.obs}"
            f"\nv={self.value:.2f}"
            f"\nn={self.visits}"
            f"\n|B|={self.belief.size()}"
            f"\npi={self.policy_str()}"
        )

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}: N{self.nid} o={self.obs} "
            f"v={self.value:.2f} n={self.visits}>"
        )


class ActionNode(Node):
    """An action node in the search tree."""

    def __init__(
        self,
        parent: ObsNode,
        action: M.ActType,
        prob: float,
        t: int,
        init_value: float = 0.0,
        init_visits: float = 0.0,
        init_total_value: float = 0.0,
    ):
        super().__init__()
        self.parent = parent
        self.action = action
        self.prob = prob
        self.t = t
        self.value = init_value
        self.visits = init_visits
        self.total_value = init_total_value
        # for calculating rolling variance
        self.agg = 0
        self.var = 0
        self.children: List[ObsNode] = []

    def get_child(self, obs: M.ObsType) -> ObsNode:
        """Get child obs node matching given observation."""
        for obs_node in self.children:
            if obs_node.obs == obs:
                return obs_node
        raise AssertionError(f"ActionNode {str(self)} has no child node for {obs=}")

    def has_child(self, obs: M.ObsType) -> bool:
        """Check if node has a child node matching history."""
        return any(obs_node.obs == obs for obs_node in self.children)

    def update(self, new_value: float):
        """Update action node statistics.

        Uses Welford's online algorithm for efficiently tracking the rolling
        variance.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        """
        self.visits += 1
        self.total_value += new_value
        delta = new_value - self.value
        self.value += delta / self.visits
        delta2 = new_value - self.value
        self.agg += delta * delta2

    @property
    def variance(self) -> float:
        """Get the variance of the value estimate for this node."""
        if self.visits == 0:
            return 0
        return self.agg / self.visits

    def __str__(self):
        return (
            f"N{self.nid}"
            f"\na={self.action}"
            f"\nt={self.t}"
            f"\nn={self.visits}"
            f"\nv={self.value:.2f}"
            f"\nw={self.total_value:.2f}"
            f"\ns2={self.variance:.2f}"
            f"\np={self.prob:.2f}"
        )

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}: "
            f"N{self.nid} "
            f"a={self.action} "
            f"t={self.t} "
            f"n={self.visits} "
            f"v={self.value:.2f} "
            f"w={self.total_value:.2f} "
            f"s2={self.variance:.2f} "
            f"p={self.prob:.2f}>"
        )


class NullNode(ActionNode):
    """The Null Node which is the parent of the root node of the tree.

    This class is mainly defined for typechecking convenience...
    """

    def __init__(self):
        super().__init__(None, action=None, prob=1.0, t=0)
