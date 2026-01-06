import ipdb
import jax.numpy as jnp
import numpy as np
from dvi.dynamics.discrete import DiscreteDyn
from loguru import logger

from valtr.reachability import DAGAvoid, DagBuilder, DAGId, DAGMaxN, DAGMinN, DAGReachAvoid, has_temporal_children


class MinTimeRollout:
    def __init__(
        self,
        dyn: DiscreteDyn,
        dag: DagBuilder,
        dag_root: DAGId,
        dict_vars: dict[int, jnp.ndarray],
        dict_actions: dict[int, jnp.ndarray],
    ):
        self.dyn = dyn
        self.dag = dag
        self.dag_root = dag_root
        self.dict_vars = dict_vars
        self.dict_actions = dict_actions

    def rollout(self, start_state: int, max_steps: int = 10):
        state = start_state
        cur_node_id = self.dag_root
        dag_nodes = self.dag.nodes

        Tp1_states = [state]
        T_actions = []

        for kk in range(max_steps):
            node = dag_nodes[cur_node_id]
            current_value = self.dict_vars[cur_node_id][state]

            # Traverse the tree until we reach a temporal operator.
            while True:
                match node:
                    case DAGMinN(args=args):
                        # Decide which child to go to based on which is the minimum.
                        arg_values = [self.dict_vars[arg][state] for arg in args]
                        cur_node_id = args[int(np.argmin(arg_values))]
                    case DAGMaxN(args=args):
                        # Decide which child to go to based on which is the maximum.
                        arg_values = [self.dict_vars[arg][state] for arg in args]
                        cur_node_id = args[int(np.argmax(arg_values))]
                    case DAGReachAvoid(reach=reach_idx, avoid=avoid):
                        reach_node = dag_nodes[reach_idx]
                        if has_temporal_children(reach_idx, self.dag):
                            # It's either just a min, or a max of mins.
                            # If it's the latter, identify which branch to take.

                            if isinstance(reach_node, DAGMaxN):
                                min_values = [self.dict_vars[idx][state] for idx in reach_node.args]
                                branch = int(np.argmax(min_values))
                                logger.info("min values: {}".format(min_values))
                                reach_min_idx = reach_node.args[branch]
                            else:
                                assert isinstance(reach_node, DAGMinN)
                                reach_min_idx = reach_idx

                            reach_min_node = dag_nodes[reach_min_idx]
                            assert isinstance(reach_min_node, DAGMinN)

                            # There should only be one temporal child under the min node.
                            temporal_idxs = [
                                child_idx for child_idx in reach_min_node.args if dag_nodes[child_idx].is_temporal()
                            ]
                            assert len(temporal_idxs) == 1
                            temporal_idx = temporal_idxs[0]

                            non_temporal_idxs = [
                                child_idx for child_idx in reach_min_node.args if not dag_nodes[child_idx].is_temporal()
                            ]

                            non_temporal_values = np.array([self.dict_vars[ii][state] for ii in non_temporal_idxs])
                            non_temporal_value = np.min(non_temporal_values)
                            temporal_value = self.dict_vars[temporal_idx][state]

                            logger.info(
                                "Cur Node: {} | Min Node: {} | nontemp_idxs: {} | temp_idxs: {} | nontemp: {} | temp: {} | cur: {}".format(
                                    cur_node_id,
                                    reach_min_idx,
                                    non_temporal_idxs,
                                    temporal_idxs,
                                    non_temporal_value,
                                    temporal_value,
                                    current_value,
                                )
                            )

                            if (non_temporal_value < temporal_value) and (non_temporal_value < current_value):
                                # Stay on the current node if (non_temporal < temporal) AND (non_temporal < V)
                                break
                            else:
                                # Otherwise, go to the temporal child.
                                cur_node_id = temporal_idx
                                logger.info("Switching to temporal child: {}".format(cur_node_id))
                                break
                        else:
                            # If no temporal children, then execute the action associated with the reach node.
                            break
                    case DAGAvoid(avoid=avoid):
                        # Avoid shouldn't have any temporal children.
                        assert not has_temporal_children(avoid, self.dag)

                        # Execute the action associated with the avoid node.
                        break
                    case _:
                        raise ValueError(f"Unexpected node type: {node}")

            # At this point, we have reached a temporal operator. Choose the action associated with this node.
            action_dict = self.dict_actions[cur_node_id]
            action = action_dict[state]

            # Apply the action to get the next state.
            state_new = self.dyn.step(state, action)

            # # If the current value >= next value, and there are no temporal children, then the trajectory has ended.
            # current_value = self.dict_vars[cur_node_id][state]
            # next_value = self.dict_vars[cur_node_id][state_new]
            # if current_value >= next_value and not has_temporal_children(cur_node_id, self.dag):
            #     break

            state = state_new
            Tp1_states.append(state)
            T_actions.append(action)

        Tp1_states = np.array(Tp1_states)
        T_actions = np.array(T_actions)

        return Tp1_states, T_actions
