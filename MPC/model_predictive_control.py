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
            own_target: np.ndarray,
            max_speed: float,
            desired_speed: float
    ):
        self.time_step = time_step
        self.visible_agents = visible_agents

        self.own_target = casadi.SX(own_target)
        # self.targets = casadi.SX(np.array([[2, 1]]))

        self.model = self.create_model()
        self.controller = self.create_controller(max_speed, desired_speed)
        self.estimator = do_mpc.estimator.StateFeedback(self.model)
        self.simulator = self.create_simulator()

        self.state = None

    def create_model(self):
        model = do_mpc.model.Model('continuous')
        # states
        # pos = model.set_variable('_x', 'pos', shape=(visible_agents, 2))
        # vel = model.set_variable('_x', 'vel', shape=(visible_agents, 2))
        own_pos = []
        own_vel = []
        for agent_idx in range(self.visible_agents):
            own_pos.append(model.set_variable('_x', f'own_pos{agent_idx}', shape=(2, 1)))
            # controls
            own_vel.append(model.set_variable('_u', f'own_vel{agent_idx}', shape=(2, 1)))

        # uncertainty parameters
        # targ = model.set_variable('_p', 'targ', shape=(visible_agents, 2))
        # own_targ = model.set_variable('_p', 'own_targ', shape=(2, 1))
        # uncertainty time varying parameters
        # velocities_uncertainty = model.set_variable('_tvp', 'velocities_uncertainty')

        # set state equations
        # model.set_rhs('pos', pos + vel * self.time_step)
        # assert pos.shape == (pos + vel * self.time_step).shape

        # model.set_rhs('vel', (self.targets - pos) * 0.5)
        # assert vel.shape == ((self.targets - pos) * 0.5).shape
        for agent_idx in range(self.visible_agents):
            model.set_rhs(f'own_pos{agent_idx}', own_vel[agent_idx] * self.time_step)
            assert own_pos[agent_idx].shape == (own_vel[agent_idx] * self.time_step).shape

        model.setup()
        return model

    def create_controller(self, max_speed, desired_speed):
        m = self.model
        mpc = do_mpc.controller.MPC(m)
        # controller parameters
        params = {
            'n_horizon': 20,
            'n_robust': 1,
            't_step': self.time_step
        }
        mpc.set_param(**params)
        # objective = sum_over_time(lagrange_term + r_term) + meyer_term

        # lterm=casadi.norm_2(1 / (m.x['own_pos'] - m.x['pos'])),
        lterm = casadi.SX(0)  # casadi.norm_2(1 / (m.x['own_pos'] - casadi.SX(np.array([5, 5]))))
        # for a, b in combinations(list(range(self.visible_agents))):
        #     lterm +=

        mterm = casadi.SX(0)
        for agent_idx in range(self.visible_agents):
            targ = self.own_target[agent_idx, :].T
            mterm += casadi.norm_2(targ - m.x[f'own_pos{agent_idx}'])

        mpc.set_objective(
            lterm=lterm,
            mterm=mterm
        )
        rterms = {f'own_vel{idx}': 1e-4 for idx in range(self.visible_agents)}
        mpc.set_rterm(**rterms)  # input regularization

        for agent_idx in range(self.visible_agents):
            mpc.set_nl_cons(
                f'speed_constraint_upper{agent_idx}',
                casadi.sum2(m.u[f'own_vel{agent_idx}']**2),
                ub=desired_speed**2,
                soft_constraint=True,
                maximum_violation=max_speed**2,
                penalty_term_cons=1e-2
            )
        for a, b in combinations(list(range(self.visible_agents)), 2):
            mpc.set_nl_cons(
                f'collision_avoidance_constraint{a}/{b}',
                -casadi.sum2((m.x[f'own_pos{a}'] - m.x[f'own_pos{b}']) ** 2),
                ub=-(10 ** 2),
                # soft_constraint=True,
                # maximum_violation=max_speed ** 2,
                # penalty_term_cons=1e-2
            )
            # mpc.set_nl_cons(
            #     f'speed_constraint_lower{agent_idx}',
            #     -m.u[f'own_vel{agent_idx}'],
            #     ub=desired_speed,
            #     soft_constraint=True,
            #     maximum_violation=max_speed,
            #     penalty_term_cons=0.05
            # )

        # mpc.bounds['upper', '_u', 'own_vel'] = 50
        # mpc.bounds['lower', '_u', 'own_vel'] = -50

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

    def init_control_loop(self, own_pos):
        x0 = self.controller.x0
        x0['own_pos0'] = own_pos[0]
        x0['own_pos1'] = own_pos[1]
        x0['own_pos2'] = own_pos[2]

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
        visible_agents=3,
        time_step=0.1,
        own_target=np.array([[0, 1], [150, 6], [10, 100]]),
        max_speed=50.,
        desired_speed=40.
    )
    own_pos = np.array([[200, 250], [100, 100], [200, 100]])
    mpc_model.init_control_loop(own_pos=own_pos)

    # mpc_graphics = do_mpc.graphics.Graphics(mpc_model.controller.data)
    # sim_graphics = do_mpc.graphics.Graphics(mpc_model.simulator.data)

    # visualizer = DataVisualizer()
    # visualizer.set_graphics_data(sim_graphics=sim_graphics, mpc_graphics=mpc_graphics)
    # visualizer.create_figure()

    t = 0
    while t < 100:  # and casadi.norm_2(mpc_model.state - mpc_model.own_target) > 0.1:
        mpc_model.update_control_loop()
        # visualizer.update_plots(mpc_model.controller.data)
        t += 1

    # visualizer.animate(T)

    rcParams['axes.grid'] = True
    rcParams['font.size'] = 18

    fig, ax, graphics = do_mpc.graphics.default_plot(mpc_model.controller.data, figsize=(16, 9))
    graphics.plot_results()
    graphics.reset_axes()
    plt.savefig('data.png')
