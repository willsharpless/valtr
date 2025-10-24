
import jax.numpy as jnp
import hj_reachability as hj
import hj_reachability.dynamics as dynamics

FLOW_RIGHT = 0.25
FLOW_UP = 0.1

class Point(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 u_bd=jnp.ones(2),
                 d_bd=jnp.ones(2),
                 control_mode="min",
                 disturbance_mode="max",
                 input_shape="ball"):
        self.dim = 2
        if input_shape == "box":
            control_space = hj.sets.Box(-u_bd * jnp.ones(2), u_bd * jnp.ones(2))
            disturbance_space  = hj.sets.Box(-d_bd * jnp.ones(2), d_bd * jnp.ones(2))
        else:
            control_space = hj.sets.Ball(center=jnp.zeros(2), radius=u_bd)
            disturbance_space = hj.sets.Ball(center=jnp.zeros(2), radius=d_bd)

        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        return FLOW_RIGHT * jnp.array([-1., 0.]) + FLOW_UP * jnp.array([0., -1.])

    def control_jacobian(self, state, time):
        return jnp.eye(self.dim)

    def disturbance_jacobian(self, state, time):
        return jnp.eye(self.dim)
    
class PointN(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 u_bd=0.0,
                 d_bd=0.0,
                 N=1,
                 control_mode="min",
                 disturbance_mode="max",
                 input_shape="ball"):
        self.N = N
        self.dim = 2*N
        self.eo = jnp.ravel(jnp.column_stack((jnp.ones(N), jnp.zeros(N))))
        self.oe = jnp.ravel(jnp.column_stack((jnp.zeros(N), jnp.ones(N))))
        if input_shape == "box":
            control_space = hj.sets.Box(-u_bd * jnp.ones(2*N), u_bd * jnp.ones(2*N))
            disturbance_space  = hj.sets.Box(-d_bd * jnp.ones(2*N), d_bd * jnp.ones(2*N))
        else:
            control_space = hj.sets.Ball(center=jnp.zeros(2*N), radius=u_bd)
            disturbance_space = hj.sets.Ball(center=jnp.zeros(2*N), radius=d_bd)

        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        return - FLOW_RIGHT * self.eo - FLOW_UP * self.oe
        # return -FLOW_RIGHT

    def control_jacobian(self, state, time):
        return jnp.eye(self.dim)

    def disturbance_jacobian(self, state, time):
        return jnp.eye(self.dim)