import ipdb
import jax.numpy as jnp
import numpy as np
import tqdm
from dvi.dynamics.discrete import ActionInt, DiscreteDyn
from loguru import logger

from dvi.dynamics.gridworld import GridWorld
from dvi.dynamics.gridworld_ma import GridWorldMA

from valtr.reachability import (DAGAvoid, DAGGUMinN, DAGGUSingle, DAGId, DAGMaxN, DAGMinN, DAGNode, DAGReach,
                                DAGReachAvoid, has_temporal_children)

from valtr.safety_filter import SafetyFilter

class Policy:
    def __call__(self, states: list[int]) -> ActionInt:
        raise NotImplementedError

class RandomPolicy(Policy):
    """Policy which takes random actions (mostly for debugging)"""
    def __init__(self, dyn: DiscreteDyn, rng: np.random.Generator | None = None):
        self.dyn = dyn
        self.rng = rng if rng else np.random.default_rng()

    def __call__(self, state_history: list[int]) -> ActionInt:
        del state_history
        return int(self.rng.integers(self.dyn.n_actions))

class GridSearchPolicy(Policy):
    """Policy for GridWorld dynamics that prefers least-visited successor states."""
    def __init__(self, dyn: GridWorld, rng: np.random.Generator | None = None):
        self.dyn = dyn
        self.rng = rng if rng else np.random.default_rng()
        self.visit_counts: dict[int, int] = {}

    def __call__(self, state_history: list[int]) -> ActionInt:
        s_cur = state_history[-1]
        self.visit_counts[s_cur] = self.visit_counts.get(s_cur, 0) + 1

        # Get all possible next states for each action.
        s_nexts = [int(self.dyn.step(s_cur, a)) for a in range(self.dyn.n_actions)]

        # Select the action which leads to the least visited next state.
        visit_counts = [self.visit_counts.get(s_next, 0) for s_next in s_nexts]
        min_visit_count = min(visit_counts)
        candidate_actions = [a for a, count in enumerate(visit_counts) if count == min_visit_count]

        return int(self.rng.choice(candidate_actions))


class GridSearchPolicyMA(Policy):
    """Multi-agent grid search that tracks visits to per-agent base-grid states."""

    def __init__(self, dyn: GridWorldMA, rng: np.random.Generator | None = None):
        self.dyn = dyn
        self.rng = rng if rng else np.random.default_rng()
        self.visit_counts: dict[int, int] = {}

    def _record_joint_state_visits(self, joint_state: int) -> None:
        agent_states = self.dyn.decode_joint_state(joint_state, which=np)
        for agent_state in agent_states:
            agent_state = int(agent_state)
            self.visit_counts[agent_state] = self.visit_counts.get(agent_state, 0) + 1

    def __call__(self, state_history: list[int]) -> ActionInt:
        s_cur = state_history[-1]
        self._record_joint_state_visits(s_cur)

        action_scores: list[tuple[int, int]] = []
        for action in range(self.dyn.n_actions):
            s_next = int(self.dyn.step(s_cur, action, which=np))
            next_agent_states = self.dyn.decode_joint_state(s_next, which=np)
            next_visit_counts = [self.visit_counts.get(int(agent_state), 0) for agent_state in next_agent_states]

            # Prefer joint actions that send some agent to a rarely visited state,
            # then break ties by the total visitation across all agents' next states.
            action_scores.append((min(next_visit_counts), sum(next_visit_counts)))

        best_score = min(action_scores)
        candidate_actions = [action for action, score in enumerate(action_scores) if score == best_score]
        return int(self.rng.choice(candidate_actions))

def get_preset_nominal_policy(preset: str, dyn: DiscreteDyn) -> Policy:
    if preset == "random":
        return RandomPolicy(dyn)
    elif preset == "grid_search":
        if isinstance(dyn, GridWorldMA):
            return GridSearchPolicyMA(dyn)
        assert isinstance(dyn, GridWorld), "GridSearchPolicy requires GridWorld or GridWorldMA dynamics"
        return GridSearchPolicy(dyn)
    else:
        raise ValueError(f"Unknown preset nominal policy: {preset}")

class FilteredRollout:
    def __init__(
        self,
        dyn: DiscreteDyn,
        dag_nodes: list[DAGNode],
        dag_root: DAGId,
        dict_vars: dict[int, jnp.ndarray],
        dict_actions: dict[int, jnp.ndarray],
        dict_GU_vars: dict[int, list[jnp.ndarray]],
        dict_GU_actions: dict[int, list[jnp.ndarray]],
        nominal_policy: Policy | None = None,
        preset_nominal: str | None = "random",
    ):
        self.dyn = dyn
        self.dag_nodes = dag_nodes
        self.dag_root = dag_root
        self.dict_vars = dict_vars
        self.dict_actions = dict_actions

        self.dict_GU_vars = dict_GU_vars
        self.dict_GU_actions = dict_GU_actions

        self.nominal_policy = nominal_policy if nominal_policy else get_preset_nominal_policy(preset_nominal, dyn)
        self.filter = SafetyFilter(dyn, dag_nodes, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions)

    def rollout(self, start_state: int, max_steps: int = 10, quiet: bool = False, which=jnp, filter: bool=False):
        state = start_state
        # cur_node_id = self.dag_root
        # dag_nodes = self.dag_nodes

        Tp1_states = [state]
        T_actions = []
        T_curnode_idxs = []

        for kk in tqdm.trange(max_steps):

            # if kk >= 52:
            #     logger.debug(f"kk={kk}")

            action_nominal = self.nominal_policy(Tp1_states)

            # Filter action
            action = self.filter.filter_action(state, action_nominal)

            # Apply the action to get the next state.
            state_new = self.dyn.step(state, action, which=which)

            state = state_new
            Tp1_states.append(state)
            T_actions.append(action)
            T_curnode_idxs.append(self.filter.cur_node_id)

        Tp1_states = np.array(Tp1_states)
        T_actions = np.array(T_actions)
        T_curnode_idxs = np.array(T_curnode_idxs)

        return Tp1_states, T_actions, T_curnode_idxs
