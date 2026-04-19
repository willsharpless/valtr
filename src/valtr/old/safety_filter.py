from typing import Protocol

import ipdb
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from dvi.dynamics.discrete import ActionInt, DiscreteDyn, StateInt
from dvi.dynamics.gridworld import GridWorld
from dvi.dynamics.gridworld_ma_timed import GridWorldMATimed
from loguru import logger

from .reachability import (DAGAvoid, DAGGUMinN, DAGGUSingle, DAGId, DAGMaxN, DAGMinN, DAGNode, DAGReach,
                                DAGReachAvoid, has_temporal_children)


class PreferenceFn(Protocol):
    """Return the costs/preference for different actions given the nominal action."""

    def __call__(self, state: StateInt, a_nom: ActionInt) -> np.ndarray: ...


class SafetyFilter:
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
        self.dyn = dyn
        self.dag_nodes = dag_nodes
        self.dag_root = dag_root
        self.dict_vars = dict_vars
        self.dict_actions = dict_actions

        self.dict_GU_vars = dict_GU_vars
        self.dict_GU_actions = dict_GU_actions
        # ---------------
        self.cur_node_id = self.dag_root
        self.cycle_start_GU_idx = None
        self.GU_index_dict: dict[DAGId, int] = {}
        self.kk = -1

    def filter_action(
        self, state: StateInt, a_nom: ActionInt, preference_fn: PreferenceFn | None = None, which=jnp
    ) -> ActionInt:
        self.kk += 1
        dag_nodes = self.dag_nodes

        state_next_nom = self.dyn.step(state, a_nom)

        current_value = self.dict_vars[self.cur_node_id][state]
        logger.debug(
            f"Start at node {self.cur_node_id} ({self.dag_nodes[self.cur_node_id]}). Current value: {current_value}"
        )

        # Evaluate the Q and get the optimal policy.
        while True:
            node = dag_nodes[self.cur_node_id]
            current_value = self.dict_vars[self.cur_node_id][state]
            action_dict = None

            if current_value < 0:
                logger.error("current value became negative!")
                ipdb.set_trace()

            match node:
                case DAGMinN(args=args):
                    # Decide which child to go to based on which is the minimum.
                    arg_values = [self.dict_vars[arg][state] for arg in args]
                    self.cur_node_id = args[int(np.argmin(arg_values))]
                case DAGMaxN(args=args):
                    # Decide which child to go to based on which is the maximum.
                    arg_values = [self.dict_vars[arg][state] for arg in args]
                    self.cur_node_id = args[int(np.argmax(arg_values))]
                case DAGReachAvoid(reach=reach_idx, avoid=_) | DAGReach(reach=reach_idx):
                    reach_node = dag_nodes[reach_idx]
                    if not has_temporal_children(reach_idx, self.dag_nodes):
                        rew = self.dict_vars[reach_idx][state]
                        if rew >= current_value:
                            # If we have reached the maximum, then we are done.
                            is_done = True

                        # If no temporal children, then execute the action associated with the reach node.
                        break

                    # Switch only if r >= current_value.
                    should_switch = self.dict_vars[reach_idx][state] >= current_value
                    if not should_switch:
                        # If we shouldn't switch, then execute the action associated with the current node.
                        break

                    # We can switch. Switch to the temporal node associated with the highest child of the max.
                    # It's either just a min, or a max of mins.
                    # If it's the latter, identify which branch to take.
                    if isinstance(reach_node, DAGMaxN):
                        min_values = [self.dict_vars[idx][state] for idx in reach_node.args]
                        branch = int(np.argmax(min_values))
                        # logger.info("min values: {}".format(min_values))
                        reach_min_idx = reach_node.args[branch]
                    else:
                        assert isinstance(reach_node, (DAGMinN, DAGGUMinN))
                        reach_min_idx = reach_idx

                    reach_min_node = dag_nodes[reach_min_idx]

                    # We now have either min( ..., temporal ), or just temporal.
                    if isinstance(reach_min_node, DAGMinN):
                        children = reach_min_node.args
                    else:
                        children = [reach_min_idx]

                    # Find the temporal node among the children.
                    temporal_idxs = [
                        child_idx
                        for child_idx in children
                        if dag_nodes[child_idx].is_temporal() or isinstance(dag_nodes[child_idx], DAGGUMinN)
                    ]
                    assert len(temporal_idxs) == 1
                    temporal_idx = temporal_idxs[0]

                    # Go to the temporal child.
                    logger.info("Switching from {} to temporal child: {}".format(self.cur_node_id, temporal_idx))
                    self.cur_node_id = temporal_idx
                    continue
                case DAGGUMinN(args=args):
                    args: list[DAGId]
                    if len(args) == 1:
                        logger.info("GU with one argument")
                        # Only one argument, execute its action
                        break

                    if self.cur_node_id not in self.GU_index_dict:
                        self.GU_index_dict[self.cur_node_id] = 0

                    # GU is a "leaf node", we just need to cycle through its arguments.
                    cur_GU_index = self.GU_index_dict[self.cur_node_id]
                    GU_single_node = self.dag_nodes[args[cur_GU_index]]
                    assert isinstance(GU_single_node, DAGGUSingle)
                    cur_q_idx, cur_r_idx = GU_single_node.avoid, GU_single_node.reach
                    r_value = self.dict_vars[cur_r_idx][state]

                    # Get the value of the NEXT state for the NEXT GU arg.
                    action_dict = self.dict_GU_actions[self.cur_node_id][cur_GU_index]
                    # action_curr = action_dict[state]
                    # state_new = self.dyn.step(state, action_curr, which=which)

                    next_GU_index = (cur_GU_index + 1) % len(args)
                    # value_next_GU = self.dict_GU_vars[self.cur_node_id][next_GU_index][state_new]

                    # Switch if min(r, value_next_GU) >= current_value.
                    switch_GU_node = r_value >= current_value
                    stay_current_GU_node = not switch_GU_node

                    if current_value == -1:
                        ipdb.set_trace()

                    if stay_current_GU_node:
                        # We are currently evaluating this node.
                        break

                    # Check for cycles.
                    if self.cycle_start_GU_idx is None:
                        self.cycle_start_GU_idx = cur_GU_index
                    elif self.cycle_start_GU_idx == cur_GU_index:
                        logger.info(
                            "Cycle start at GU idx {}, detected cycle on GU args at index {}".format(
                                self.cycle_start_GU_idx, cur_GU_index
                            )
                        )
                        # We have cycled through all GU args. Execute the current action and move to next GU arg.
                        self.GU_index_dict[self.cur_node_id] = next_GU_index
                        break

                    # Otherwise, advance to the next GU node.
                    self.GU_index_dict[self.cur_node_id] = next_GU_index
                    continue

                case DAGAvoid(avoid=avoid):
                    # Avoid shouldn't have any temporal children.
                    assert not has_temporal_children(avoid, self.dag_nodes)

                    # Execute the action associated with the avoid node.
                    break
                case _:
                    raise ValueError(f"Unexpected node type: {node}")

        # End of while.

        # if self.cur_node_id == 26:
        #     assert isinstance(node, DAGReachAvoid)
        #     reach_idx = node.reach
        #     reach_node = dag_nodes[reach_idx]
        #     assert isinstance(reach_node, DAGMinN)
        #     dyn_ma: GridWorldMATimed = self.dyn
        #
        #     s_joint_, tt_ = dyn_ma.decode_timed_state(state)
        #
        #     reach_node_value = self.dict_vars[reach_idx][state]
        #     logger.warning(f"reach_node value: {reach_node_value}")
        #     ipdb.set_trace()

        # At this point, we have reached a temporal operator. Choose the action associated with this node.
        if action_dict is None:
            action_dict = self.dict_actions[self.cur_node_id]
        action_optpol = action_dict[state]

        # Safety filtering logic: if nominal action is safe, then take it. Otherwise, take the optimal action.
        value_next_nom = self.dict_vars[self.cur_node_id][state_next_nom]

        # dyn: GridWorld = self.dyn
        # decoded_state = [int(n) for n in dyn.decode_state(state, np)]
        # action_str = dyn.action_to_str(a_nom)
        # decoded_nextstate = [int(n) for n in dyn.decode_state(state_next_nom)]
        # logger.debug(f"{decoded_state} ({action_str}) -> {decoded_nextstate}. nom_value = {value_next_nom}")

        nom_is_safe = value_next_nom >= 0

        logger.debug(
            f"End at node {self.cur_node_id} ({self.dag_nodes[self.cur_node_id]}). Current value: {current_value}, next nom value: {value_next_nom}"
        )

        # if self.cur_node_id == 18 and not nom_is_safe:
        #     logger.warning("At node 18, nominal action is not safe!")
        #     ipdb.set_trace()

        if nom_is_safe:
            return a_nom

        if preference_fn is not None:

            def is_action_safe(action_) -> bool:
                state_next_ = self.dyn.step(state, action_)
                value_next_ = jnp.array(self.dict_vars[self.cur_node_id])[state_next_]
                return value_next_ >= 0

            a_actions = np.arange(self.dyn.n_actions)
            a_isactionsafe = jax.vmap(is_action_safe)(a_actions)
            a_isactionsafe = jax.device_get(a_isactionsafe)
            n_actions_safe = np.sum(a_isactionsafe)
            assert n_actions_safe >= 1, "There should be at least one safe action."

            n_costs = preference_fn(state, a_nom)
            n_costs = np.where(a_isactionsafe, n_costs, np.inf)
            action_filtered = np.argmin(n_costs)

            # if action_filtered == 0:
            #     logger.warning("Preference fn chose action 0!")
            # if self.kk == 16:
            #     logger.warning("kk={} | Nominal action: {}".format(self.kk, self.dyn.action_to_str(a_nom)))
            #
            #     for action_ in range(self.dyn.n_actions):
            #         if not a_isactionsafe[action_]:
            #             continue
            #         action_safe_str = self.dyn.action_to_str(action_)
            #         logger.warning(f"Safe action {action_}: {action_safe_str}, cost: {n_costs[action_]}")
            #
            #     ipdb.set_trace()

            ## more sophisticated preference strategy
            # safe_actions = np.flatnonzero(a_isactionsafe)
            # safe_costs = np.asarray(n_costs)[safe_actions]

            # safe_next_states = np.array([int(self.dyn.step(state, int(action), which=np)) for action in safe_actions], dtype=np.int32)
            # safe_next_values = np.asarray(self.dict_vars[self.cur_node_id])[safe_next_states]

            # min_cost = np.min(safe_costs)
            # tol = 1e-6
            # candidate_mask = safe_costs <= (min_cost + tol)
            # candidate_actions = safe_actions[candidate_mask]
            # candidate_values = safe_next_values[candidate_mask]

            # action_filtered = int(candidate_actions[np.argmax(candidate_values)])
            return action_filtered
        else:
            return action_optpol
