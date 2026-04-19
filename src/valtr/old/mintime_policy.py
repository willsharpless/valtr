import ipdb
import jax.numpy as jnp
import numpy as np
import tqdm
from dvi.dynamics.discrete import ActionInt, DiscreteDyn
from dvi.dynamics.gridworld_ma_timed import GridWorldMATimed
from dvi.dynamics.gridworld_timed import GridWorldTimed
from loguru import logger

from .reachability import (DAGAvoid, DAGGUMinN, DAGGUSingle, DAGId, DAGMaxN, DAGMinN, DAGNode, DAGReach,
                                DAGReachAvoid, has_temporal_children)


class MinTimePolicy:
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

    def get_action(self, state: jnp.ndarray, which=jnp, kk: int | None = None, debug: bool = False) -> tuple[ActionInt, bool]:
        dag_nodes = self.dag_nodes
        cur_node_id = self.cur_node_id

        is_done = False

        if cur_node_id == len(self.dag_nodes):
            return None, True

        if debug:
            dyn_ma = self.dyn
            state_ag1, state_ag2 = dyn_ma.decode_joint_state(state, which=np)
            state_ag1_tup = [int(n) for n in dyn_ma.base.decode_state(state_ag1)]
            state_ag2_tup = [int(n) for n in dyn_ma.base.decode_state(state_ag2)]
            current_value = self.dict_vars[cur_node_id][state]
            logger.debug("    STATE: cur_node_id={} | val={} | agent1 at {}, agent2 at {}".format(cur_node_id, current_value, state_ag1_tup, state_ag2_tup))

        # Traverse the tree until we reach a temporal operator.
        while True:
            node = dag_nodes[cur_node_id]
            current_value = self.dict_vars[cur_node_id][state]
            action_dict = None

            if current_value < 0:
                logger.warning("Negative value at node {}: {}".format(cur_node_id, current_value))
                ipdb.set_trace()

            match node:
                case DAGMinN(args=args):
                    # Decide which child to go to based on which is the minimum.
                    arg_values = [self.dict_vars[arg][state] for arg in args]
                    cur_node_id = args[int(np.argmin(arg_values))]
                case DAGMaxN(args=args):
                    # Decide which child to go to based on which is the maximum.
                    arg_values = [self.dict_vars[arg][state] for arg in args]
                    cur_node_id = args[int(np.argmax(arg_values))]
                case DAGReachAvoid(reach=reach_idx, avoid=_) | DAGReach(reach=reach_idx):
                    reach_node = dag_nodes[reach_idx]
                    if not has_temporal_children(reach_idx, self.dag_nodes):
                        rew = self.dict_vars[reach_idx][state]
                        if rew >= current_value:
                            # If we have reached the maximum, then we are done.
                            is_done = True
                            logger.debug("Done! Reached node {} with value {} >= current value {}".format(reach_idx, rew, current_value))
                            # ipdb.set_trace()

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
                    logger.info("Switching from {} to temporal child: {}".format(cur_node_id, temporal_idx))
                    cur_node_id = temporal_idx
                    continue

                case DAGGUMinN(args=args):
                    args: list[DAGId]
                    if len(args) == 1:
                        logger.info("GU with one argument")
                        # Only one argument, execute its action
                        break

                    if cur_node_id not in self.GU_index_dict:
                        self.GU_index_dict[cur_node_id] = 0

                    # GU is a "leaf node", we just need to cycle through its arguments.
                    cur_GU_index = self.GU_index_dict[cur_node_id]
                    GU_single_node = self.dag_nodes[args[cur_GU_index]]
                    assert isinstance(GU_single_node, DAGGUSingle)
                    cur_q_idx, cur_r_idx = GU_single_node.avoid, GU_single_node.reach
                    r_value = self.dict_vars[cur_r_idx][state]

                    # Get the value of the NEXT state for the NEXT GU arg.
                    action_dict = self.dict_GU_actions[cur_node_id][cur_GU_index]
                    action_curr = action_dict[state]
                    state_new = self.dyn.step(state, action_curr, which=which)

                    next_GU_index = (cur_GU_index + 1) % len(args)
                    value_next_GU = self.dict_GU_vars[cur_node_id][next_GU_index][state_new]

                    # Switch if min(r, value_next_GU) >= current_value.
                    # stay_current_GU_node = (r_value < value_next_GU) and (r_value < current_value)
                    switch_GU_node = r_value >= current_value
                    stay_current_GU_node = not switch_GU_node

                    # If all three values are equal, then it means that:
                    # 1) We have reached the highest possible value for this reach target
                    # 2) We can reach the next GU arg equally well.
                    # It is possible that all GU args have the same value, in which case this will cycle.
                    # If it cycles, execute the current action and then move to the next GU arg (upon which the same
                    # thing will happen again).
                    # same_value = (r_value == value_next_GU) and (r_value == current_value)
                    action_curr_str = self.dyn.action_to_str(action_curr)

                    # if isinstance(self.dyn, GridWorldMATimed) or isinstance(self.dyn, GridWorldTimed):
                    #     state_str = self.dyn.decode_timed_state(state, which=np)
                    # else:
                    #     state_str = ""
                    #
                    # logger.info(
                    #     "kk={} | state: {} | Cur Node: {} | Cur GU idx: {} | r_value: {} | value_next_GU: {} | current_value: {} | "
                    #     "stay: {} | action_curr: {}".format(
                    #         kk,
                    #         state_str,
                    #         cur_node_id,
                    #         cur_GU_index,
                    #         r_value,
                    #         value_next_GU,
                    #         current_value,
                    #         stay_current_GU_node,
                    #         action_curr_str,
                    #     )
                    # )

                    if current_value == -1:
                        ipdb.set_trace()

                    # if same_value:
                    if not stay_current_GU_node:
                        if self.cycle_start_GU_idx is None:
                            self.cycle_start_GU_idx = cur_GU_index
                        elif self.cycle_start_GU_idx == cur_GU_index:
                            logger.info(
                                "Cycle start at GU idx {}, detected cycle on GU args at index {}".format(
                                    self.cycle_start_GU_idx, cur_GU_index
                                )
                            )
                            # We have cycled through all GU args. Execute the current action and move to next GU arg.
                            self.GU_index_dict[cur_node_id] = next_GU_index
                            break

                    if stay_current_GU_node:
                        # Stay on the current GU arg if (r_value < value_next_GU) and (r_value < current_value)
                        # logger.info("Taking action for GU index {} | {}".format(cur_GU_index, action_curr_text))
                        break
                    else:
                        # Otherwise, advance to the next GU arg.
                        self.GU_index_dict[cur_node_id] = next_GU_index
                        # logger.info("Moving to next GU arg: {}".format(next_GU_index))
                        continue

                case DAGAvoid(avoid=avoid):
                    # Avoid shouldn't have any temporal children.
                    assert not has_temporal_children(avoid, self.dag_nodes)

                    # Execute the action associated with the avoid node.
                    break
                case _:
                    raise ValueError(f"Unexpected node type: {node}")

        # End of while True
        # At this point, we have reached a temporal operator. Choose the action associated with this node.
        if action_dict is None:
            action_dict = self.dict_actions[cur_node_id]
        action = action_dict[state]

        if debug:
            dyn_ma = self.dyn
            state_ag1, state_ag2 = dyn_ma.decode_joint_state(state, which=np)
            state_ag1_tup = [int(n) for n in dyn_ma.base.decode_state(state_ag1)]
            state_ag2_tup = [int(n) for n in dyn_ma.base.decode_state(state_ag2)]
            logger.debug("    > STATE: cur_node_id={} | action={} | agent1 at {}, agent2 at {}".format(cur_node_id, dyn_ma.action_to_str(action), state_ag1_tup, state_ag2_tup))

        if is_done:
            cur_node_id = len(self.dag_nodes)

        # Make sure to update cur_node_id.
        self.cur_node_id = cur_node_id

        return action, is_done
