import ipdb
import jax.numpy as jnp
import numpy as np
from dvi.dynamics.gridworld import GridWorld


def parse_rooms(s: str):
    s = s.strip()

    # Figure out how many rows and columns.
    lines = s.split("\n")
    height = len(lines)
    width = len(lines[0])

    # For each unique character, create an entry in the dict.
    d = {}
    for ii, l in enumerate(lines):
        assert len(l) == width

        for jj, c in enumerate(l):
            if c not in d:
                d[c] = np.zeros((height, width), dtype=bool)

            d[c][ii, jj] = True

    shape = (height, width)
    drift_fn = None
    dyn = GridWorld(shape, drift_fn)

    return dyn, d


class GridWorldDriftFn:
    def __init__(self, d: dict[str, np.ndarray], force: bool = False):
        self.d = d
        self.force = force

    def __call__(self, state: jnp.ndarray, delta: jnp.ndarray, which=jnp):
        d = self.d
        force = self.force
        y, x = state
        l_only = which.array(d["<"])[y, x]  # bool
        r_only = which.array(d[">"])[y, x]  # bool
        u_only = which.array(d["^"])[y, x]  # bool

        if "v" in d:
            d_only = which.array(d["v"])[y, x]  # bool
        else:
            d_only = np.array(False)

        delta_x = which.where(l_only, -1, which.where(r_only, 1, delta[0]))
        delta_y = which.where(u_only, -1, which.where(d_only, 1, delta[1]))

        if force:
            # In l_only or r_only, delta_y = 0. Similarly, in u_only or d_only, delta_x = 0.
            delta_y = which.where(l_only | r_only, 0, delta_y)
            delta_x = which.where(u_only | d_only, 0, delta_x)

        if isinstance(delta, jnp.ndarray):
            delta = delta.at[1].set(delta_x).at[0].set(delta_y)
        else:
            assert isinstance(delta, np.ndarray)
            delta[1] = delta_x
            delta[0] = delta_y

        return state + delta


# def get_drift_fn(d: dict[str, np.ndarray], force: bool = False):
#     """
#     :param d:
#     :param force: If true, then zero out the movement in the other direction.
#     :return:
#     """
#
#     def drift_fn(state: jnp.ndarray, delta: jnp.ndarray):
#         # If state is on <, then only allow left movement.
#         y, x = state
#         l_only = jnp.array(d["<"])[y, x]  # bool
#         r_only = jnp.array(d[">"])[y, x]  # bool
#         u_only = jnp.array(d["^"])[y, x]  # bool
#
#         if "v" in d:
#             d_only = jnp.array(d["v"])[y, x]  # bool
#         else:
#             d_only = np.array(False)
#
#         delta_x = jnp.where(l_only, -1, jnp.where(r_only, 1, delta[0]))
#         delta_y = jnp.where(u_only, -1, jnp.where(d_only, 1, delta[1]))
#
#         if force:
#             # In l_only or r_only, delta_y = 0. Similarly, in u_only or d_only, delta_x = 0.
#             delta_y = jnp.where(l_only | r_only, 0, delta_y)
#             delta_x = jnp.where(u_only | d_only, 0, delta_x)
#
#         delta = delta.at[1].set(delta_x).at[0].set(delta_y)
#
#         return state + delta
#
#     return drift_fn
