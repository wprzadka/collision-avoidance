import do_mpc
import casadi
import numpy as np
from data_visualizer import DataVisualizer
from matplotlib import rcParams
import matplotlib.pyplot as plt


class ModelPredictiveControl:

    def __init__(self, visible_agents, time_step):
        self.time_step = time_step

        self.own_target = casadi.SX(np.array([100, 500]))  #.reshape((2, 1)))
        self.targets = casadi.SX(np.array([[2, 1]]))

        self.model = self.create_model(visible_agents)
        self.controller = self.create_controller()
        self.estimator = do_mpc.estimator.StateFeedback(self.model)
        self.simulator = self.create_simulator()

        self.state = None

    def create_model(self, visible_agents):
        model = do_mpc.model.Model('continuous')
        # states
        # pos = model.set_variable('_x', 'pos', shape=(visible_agents, 2))
        # vel = model.set_variable('_x', 'vel', shape=(visible_agents, 2))
        own_pos = model.set_variable('_x', 'own_pos', shape=(2, 1))
        # controls
        own_vel = model.set_variable('_u', 'own_vel', shape=(2, 1))

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

        model.set_rhs('own_pos', own_vel * self.time_step)
        assert own_pos.shape == (own_vel * self.time_step).shape

        model.setup()
        return model

    def create_controller(self):
        m = self.model
        mpc = do_mpc.controller.MPC(m)
        # controller parameters
        params = {
            'n_horizon': 4,
            'n_robust': 1,
            't_step': 0.1
        }
        mpc.set_param(**params)
        # objective = sum_over_time(lagrange_term + r_term) + meyer_term
        mpc.set_objective(
            # lterm=casadi.norm_2(1 / (m.x['own_pos'] - m.x['pos'])),
            lterm=casadi.norm_2(1 / (m.x['own_pos'] - casadi.SX(np.array([5, 5])))),
            mterm=casadi.sum2(casadi.norm_2(self.own_target - m.x['own_pos']))
        )
        mpc.set_rterm(own_vel=000.1)  # input regularization

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

    def init_control_loop(self, pos, vel, own_pos):
        x0 = self.controller.x0
        # x0['pos'] = pos
        # x0['vel'] = vel
        x0['own_pos'] = own_pos

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

    mpc_model = ModelPredictiveControl(1, 0.1)
    pos = np.array([100, 100])
    vel = np.zeros(2)
    own_pos = np.array([200, 250])
    mpc_model.init_control_loop(pos=pos, vel=vel, own_pos=own_pos)

    # mpc_graphics = do_mpc.graphics.Graphics(mpc_model.controller.data)
    # sim_graphics = do_mpc.graphics.Graphics(mpc_model.simulator.data)

    # visualizer = DataVisualizer()
    # visualizer.set_graphics_data(sim_graphics=sim_graphics, mpc_graphics=mpc_graphics)
    # visualizer.create_figure()

    t = 0
    while t < 600 and casadi.norm_2(mpc_model.state - mpc_model.own_target) > 0.1:
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