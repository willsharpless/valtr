from dvi.dynamics.discrete import ActionInt, StateInt
import ipdb
import jax.numpy as jnp
import numpy as np
import tqdm
from dvi.dynamics.discrete import DiscreteDyn
from dvi.dynamics.gridworld import GridWorld
from loguru import logger

from valtr.reachability import (DAGAvoid, DAGGUMinN, DAGGUSingle, DAGId, DAGMaxN, DAGMinN, DAGNode, DAGReach,
                                DAGReachAvoid, has_temporal_children)


class SafetyFilter:
    def __init__(self,
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

    def filter_action(self, state: StateInt, a_nom: ActionInt) -> ActionInt:
        dag_nodes = self.dag_nodes

        state_next_nom = self.dyn.step(state, a_nom)

        # Evaluate the Q and get the optimal policy.
        while True:
            node = dag_nodes[self.cur_node_id]
            current_value = self.dict_vars[self.cur_node_id][state]
            action_dict = None

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
                    if has_temporal_children(reach_idx, self.dag_nodes):
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
                        if isinstance(reach_min_node, DAGGUMinN):
                            # F G F ...
                            child_value = self.dict_vars[reach_min_idx][state]

                            if child_value < current_value:
                                # If G(...) < value, then staying here gets us a better val in the future.
                                break
                            else:
                                # Go to GU.
                                self.cur_node_id = reach_min_idx

                        else:
                            assert isinstance(reach_min_node, DAGMinN)

                            # There should only be one temporal child under the min node.
                            temporal_idxs = [
                                child_idx
                                for child_idx in reach_min_node.args
                                if dag_nodes[child_idx].is_temporal() or isinstance(dag_nodes[child_idx], DAGGUMinN)
                            ]
                            assert len(temporal_idxs) == 1
                            temporal_idx = temporal_idxs[0]

                            non_temporal_idxs = [
                                child_idx
                                for child_idx in reach_min_node.args
                                if not dag_nodes[child_idx].is_temporal()
                            ]

                            non_temporal_values = np.array([self.dict_vars[ii][state] for ii in non_temporal_idxs])
                            non_temporal_value = np.min(non_temporal_values)
                            temporal_value = self.dict_vars[temporal_idx][state]

                            if (non_temporal_value < temporal_value) and (non_temporal_value < current_value):
                                # Stay on the current node if (non_temporal < temporal) AND (non_temporal < V)
                                break
                            else:
                                # Otherwise, go to the temporal child.
                                if self.cur_node_id == temporal_idx:
                                    logger.error(
                                        "cur_node_id: {}, temporal_idx: {}".format(self.cur_node_id, temporal_idx)
                                    )
                                    ipdb.set_trace()

                                self.cur_node_id = temporal_idx
                                # logger.info("Switching to temporal child: {}".format(cur_node_id))
                                continue
                    else:
                        # If no temporal children, then execute the action associated with the reach node.
                        break
                case DAGGUMinN(args=args):
                    raise NotImplementedError("")
                    args: list[DAGId]
                    if len(args) == 1:
                        logger.info("GU with one argument")
                        # Only one argument, execute its action
                        break

                    if cur_node_id not in GU_index_dict:
                        GU_index_dict[cur_node_id] = 0

                    # GU is a "leaf node", we just need to cycle through its arguments.
                    cur_GU_index = GU_index_dict[cur_node_id]
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
                    stay_current_GU_node = (r_value < value_next_GU) and (r_value < current_value)

                    # If all three values are equal, then it means that:
                    # 1) We have reached the highest possible value for this reach target
                    # 2) We can reach the next GU arg equally well.
                    # It is possible that all GU args have the same value, in which case this will cycle.
                    # If it cycles, execute the current action and then move to the next GU arg (upon which the same
                    # thing will happen again).
                    same_value = (r_value == value_next_GU) and (r_value == current_value)
                    # action_curr_str = self.dyn.action_to_str(action_curr)

                    # logger.info(
                    #     "Cur Node: {} | Cur GU idx: {} | r_value: {} | value_next_GU: {} | current_value: {} | "
                    #     "stay: {} | action_curr: {}".format(
                    #         cur_node_id,
                    #         cur_GU_index,
                    #         r_value,
                    #         value_next_GU,
                    #         current_value,
                    #         stay_current_GU_node,
                    #         action_curr_str,
                    #     )
                    # )

                    if same_value:
                        if cycle_start_GU_idx is None:
                            cycle_start_GU_idx = cur_GU_index
                        elif cycle_start_GU_idx == cur_GU_index:
                            logger.info("Detected cycle on GU args at index {}".format(cur_GU_index))
                            # We have cycled through all GU args. Execute the current action and move to next GU arg.
                            GU_index_dict[cur_node_id] = next_GU_index
                            break

                    if stay_current_GU_node:
                        # Stay on the current GU arg if (r_value < value_next_GU) and (r_value < current_value)
                        # logger.info("Taking action for GU index {} | {}".format(cur_GU_index, action_curr_text))
                        break
                    else:
                        # Otherwise, advance to the next GU arg.
                        GU_index_dict[cur_node_id] = next_GU_index
                        # logger.info("Moving to next GU arg: {}".format(next_GU_index))
                        continue

                case DAGAvoid(avoid=avoid):
                    # Avoid shouldn't have any temporal children.
                    assert not has_temporal_children(avoid, self.dag_nodes)

                    # Execute the action associated with the avoid node.
                    break
                case _:
                    raise ValueError(f"Unexpected node type: {node}")

        # End of while.

        # At this point, we have reached a temporal operator. Choose the action associated with this node.
        if action_dict is None:
            action_dict = self.dict_actions[self.cur_node_id]
        action = action_dict[state]

        # Safety filtering logic: if nominal action is safe, then take it. Otherwise, take the optimal action.
        value_next_nom = self.dict_vars[self.cur_node_id][state_next_nom]

        dyn: GridWorld = self.dyn
        logger.debug(f"{dyn.decode_state(state)} ({dyn.action_to_str(a_nom)}) -> {dyn.decode_state(state_next_nom)}. value = {value_next_nom}")

        if value_next_nom >= 0:
            return a_nom
        else:
            return action
