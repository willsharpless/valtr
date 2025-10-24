
import jax.numpy as jnp
import hj_reachability as hj
import hj_reachability.dynamics as dynamics

class Dubins(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 u_bd=1.,
                 d_bd=0.,
                 control_mode="min",
                 disturbance_mode="max",
                 input_shape="ball"):
        self.dim = 3
        if input_shape == "box":
            control_space = hj.sets.Box(-u_bd * jnp.ones(1), u_bd * jnp.ones(1))
            disturbance_space  = hj.sets.Box(-d_bd * jnp.ones(1), d_bd * jnp.ones(1))
        else:
            control_space = hj.sets.Ball(center=jnp.zeros(1), radius=u_bd)
            disturbance_space = hj.sets.Ball(center=jnp.zeros(1), radius=d_bd)

        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        return jnp.array([[jnp.sin(state[2])], [jnp.cos(state[2])], [0.]])

    def control_jacobian(self, state, time):
        return jnp.array([[0.], [0.], [1.]])

    def disturbance_jacobian(self, state, time):
        return jnp.array([[0.], [0.], [1.]])
    

class AccelerationCurvatureCar(hj.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 max_acceleration=1.,
                 max_curvature=1.,
                 max_position_disturbance=0.25,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None):
        self.dim = 4
        if control_space is None:
            control_space = hj.sets.Box(jnp.array([-max_acceleration, -max_curvature]),
                                        jnp.array([max_acceleration, max_curvature]))
        if disturbance_space is None:
            disturbance_space = hj.sets.Ball(jnp.zeros(2), max_position_disturbance)
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        _, _, v, q = state
        return jnp.array([v * jnp.cos(q), v * jnp.sin(q), 0., 0.])

    def control_jacobian(self, state, time):
        v = state[2]
        return jnp.array([
            [0., 0.],
            [0., 0.],
            [1., 0.],
            [0., v],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [1., 0.],
            [0., 1.],
            [0., 0.],
            [0., 0.],
        ])