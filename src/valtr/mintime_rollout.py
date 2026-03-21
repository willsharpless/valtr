import ipdb
import jax.numpy as jnp
import numpy as np
import tqdm
from dvi.dynamics.discrete import DiscreteDyn
from dvi.dynamics.gridworld_ma_timed import GridWorldMATimed
from dvi.dynamics.gridworld_timed import GridWorldTimed
from loguru import logger

from valtr.mintime_policy import MinTimePolicy
from valtr.reachability import (DAGAvoid, DAGGUMinN, DAGGUSingle, DAGId, DAGMaxN, DAGMinN, DAGNode, DAGReach,
                                DAGReachAvoid, has_temporal_children)


class MinTimeRollout:
    def __init__(
        self,
        dyn: DiscreteDyn,
        dag_nodes: list[DAGNode],
        dag_root: DAGId,
        dict_vars: dict[int, jnp.ndarray],
        dict_actions: dict[int, jnp.ndarray],
        dict_GU_vars: dict[int, list[jnp.ndarray]],
        dict_GU_actions: dict[int, list[jnp.ndarray]],
    ):
        self.policy = MinTimePolicy(dyn, dag_nodes, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions)
        self.dag_root = dag_root
        self.dyn = dyn

    def rollout(self, start_state: int, max_steps: int = 10, quiet: bool = False, which=jnp, debug: bool = False):
        state = start_state

        Tp1_states = [state]
        T_actions = []
        T_curnode_idxs = []

        for kk in tqdm.trange(max_steps):
            action, isdone = self.policy.get_action(state, which=which, kk=kk, debug=debug)

            if isdone:
                break

            # Apply the action to get the next state.
            state_new = self.dyn.step(state, action, which=which)

            # # If the current value >= next value, and there are no temporal children, then the trajectory has ended.
            # current_value = self.dict_vars[cur_node_id][state]
            # next_value = self.dict_vars[cur_node_id][state_new]
            # if current_value >= next_value and not has_temporal_children(cur_node_id, self.dag):
            #     break

            state = state_new
            Tp1_states.append(state)
            T_actions.append(action)
            T_curnode_idxs.append(self.policy.cur_node_id)

        Tp1_states = np.array(Tp1_states)
        T_actions = np.array(T_actions)
        T_curnode_idxs = np.array(T_curnode_idxs)

        return Tp1_states, T_actions, T_curnode_idxs
