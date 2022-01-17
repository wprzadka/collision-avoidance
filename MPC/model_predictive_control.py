import do_mpc
import casadi
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
from itertools import combinations


class ModelPredictiveControl:

    def __init__(
            self,
            visible_agents: int,
            time_step: float,
            target: np.ndarray,
            radius: np.ndarray,
            max_speed: float,
            desired_speed: float
    ):
        self.time_step = time_step
        self.visible_agents = visible_agents
        self.target = target
        self.radius = radius

        self.model = self.create_model()
        self.controller = self.create_controller(max_speed, desired_speed)
        self.estimator = do_mpc.estimator.StateFeedback(self.model)
        self.simulator = self.create_simulator()

        self.state = None

    def create_model(self):
        model = do_mpc.model.Model('continuous')
        # states
        pos = []
        vel = []
        for agent_idx in range(self.visible_agents):
            pos.append(model.set_variable('_x', f'pos{agent_idx}', shape=(2, 1)))
            # controls
            vel.append(model.set_variable('_u', f'vel{agent_idx}', shape=(2, 1)))

        # uncertainty time varying parameters
        # velocities_uncertainty = model.set_variable('_tvp', 'velocities_uncertainty')

        # set state equations
        for agent_idx in range(self.visible_agents):
            model.set_rhs(f'pos{agent_idx}', vel[agent_idx] * self.time_step)
            assert pos[agent_idx].shape == (vel[agent_idx] * self.time_step).shape

        model.setup()
        return model

    def create_controller(self, max_speed, desired_speed):
        m = self.model
        mpc = do_mpc.controller.MPC(m)
        # controller parameters
        params = {
            'n_horizon': 20,
            # 'n_robust': 1,
            't_step': self.time_step,
            'state_discretization': 'collocation',
            'collocation_type': 'radau',
            'collocation_deg': 3,
            'collocation_ni': 1,
            # 'store_full_solution': True,
            # Use MA27 linear solver in ipopt for faster calculations:
            # 'nlpsol_opts': {'ipopt.linear_solver': 'mumps'}
        }
        mpc.set_param(**params)
        # objective = sum_over_time(lagrange_term + r_term) + meyer_term

        # keep distance to other agents
        lterm = casadi.SX(0)
        for a, b in combinations(list(range(self.visible_agents)), 2):
            lterm += 10. / casadi.norm_2(m.x[f'pos{a}'] - m.x[f'pos{b}'])

        # move over target
        mterm = casadi.SX(0)
        for agent_idx in range(self.visible_agents):
            targ = self.target[agent_idx]
            mterm += casadi.norm_2(targ - m.x[f'pos{agent_idx}'])

        mpc.set_objective(
            lterm=lterm,
            mterm=mterm
        )
        rterms = {f'vel{idx}': 1e-4 for idx in range(self.visible_agents)}
        mpc.set_rterm(**rterms)  # input regularization

        # constraint maximal speed of agents
        for agent_idx in range(self.visible_agents):
            mpc.set_nl_cons(
                f'speed_constraint_upper{agent_idx}',
                casadi.sum2(m.u[f'vel{agent_idx}'] ** 2),
                ub=desired_speed ** 2,
                soft_constraint=True,
                maximum_violation=max_speed ** 2,
                penalty_term_cons=1e-2
            )

        # constraint for collisions
        for a, b in combinations(list(range(self.visible_agents)), 2):
            rad_sum = self.radius[a] + self.radius[b]
            mpc.set_nl_cons(
                f'collision_avoidance_constraint{a}/{b}',
                -casadi.sum2((m.x[f'pos{a}'] - m.x[f'pos{b}']) ** 2),
                ub=-(rad_sum ** 2),
                soft_constraint=True,
                maximum_violation=(rad_sum / 2.) ** 2,
                penalty_term_cons=1e-1
             )

        mpc.setup()
        return mpc

    def create_simulator(self):
        simulator = do_mpc.simulator.Simulator(self.model)
        params = {
            't_step': self.time_step
        }
        simulator.set_param(
            **params
        )
        simulator.setup()
        return simulator

    def init_control_loop(self, pos):
        x0 = self.controller.x0
        for i in range(self.visible_agents):
            x0[f'pos{i}'] = pos[i]

        self.controller.x0 = x0
        self.simulator.x0 = x0
        self.estimator.x0 = x0

        self.controller.set_initial_guess()
        self.state = self.controller.x0

    def update_control_loop(self):
        assert self.state is not None
        u0 = self.controller.make_step(self.state)
        y_next = self.simulator.make_step(u0)
        self.state = self.estimator.make_step(y_next)
        return self.state


if __name__ == '__main__':

    mpc_model = ModelPredictiveControl(
        visible_agents=4,
        time_step=0.1,
        target=np.array([[0, 1], [150, 6], [10, 100], [150, 150]]),
        radius=np.full(shape=(4,), fill_value=2.),
        max_speed=50.,
        desired_speed=40.
    )
    position = np.array([[200, 250], [100, 100], [200, 100], [0, 0]])
    mpc_model.init_control_loop(pos=position)

    t = 0
    while t < 200:
        mpc_model.update_control_loop()
        t += 1

    # visualize data
    rcParams['axes.grid'] = True
    rcParams['font.size'] = 18

    fig, ax, graphics = do_mpc.graphics.default_plot(mpc_model.controller.data, figsize=(16, 9))
    graphics.plot_results()
    graphics.reset_axes()
    plt.savefig('small_MPC_plots.png')
