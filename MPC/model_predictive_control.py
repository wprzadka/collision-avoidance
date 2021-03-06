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
            desired_speed: float,
            collision_cost_weight: float = 10.,
            distance_cost_weight: float = 10.
    ):
        # problem settings
        self.time_step = time_step
        self.visible_agents = visible_agents
        self.target = casadi.SX(target)
        self.radius = radius

        # cost weights
        self.coll_cw = collision_cost_weight
        self.dist_cw = distance_cost_weight

        self.agents_positions = np.zeros((self.visible_agents, 2))
        self.agents_velocities = np.zeros((self.visible_agents, 2))

        self.model = self.create_model()

        self.controller = do_mpc.controller.MPC(self.model)
        self.create_controller(max_speed, desired_speed)

        self.estimator = do_mpc.estimator.StateFeedback(self.model)

        self.simulator = do_mpc.simulator.Simulator(self.model)
        self.create_simulator()

        self.state = None

    def create_model(self):
        model = do_mpc.model.Model('continuous')
        # states
        self_pos = model.set_variable('_x', 'self_pos', shape=(2, 1))
        self_vel = model.set_variable('_u', 'self_vel', shape=(2, 1))

        # time varying parameters
        pos = []
        vel = []
        for agent_idx in range(self.visible_agents):
            pos.append(model.set_variable('_tvp', f'pos{agent_idx}', shape=(2, 1)))
            # controls
            vel.append(model.set_variable('_tvp', f'vel{agent_idx}', shape=(2, 1)))

        # set state equations
        model.set_rhs('self_pos', self_vel * self.time_step)
        assert self_pos.shape == (self_vel * self.time_step).shape

        model.setup()
        return model

    def create_controller(self, max_speed, desired_speed):
        m = self.model
        mpc = self.controller
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
        # for a, b in combinations(list(range(self.visible_agents)), 2):
        for a in range(self.visible_agents):
            rad_sum = self.radius[a] + self.radius[0]
            lterm += self.coll_cw / (1 + casadi.exp(casadi.norm_2(m.x['self_pos'] - m.tvp[f'pos{a}']) - rad_sum))

        # move over target
        mterm = self.dist_cw * casadi.norm_2(self.target - m.x['self_pos'])

        mpc.set_objective(
            lterm=lterm,
            mterm=mterm
        )
        rterms = {'self_vel': 1e-4}
        mpc.set_rterm(**rterms)  # input regularization

        # constraint maximal speed of agents
        mpc.set_nl_cons(
            'speed_constraint',
            casadi.sum2(m.u['self_vel'] ** 2),
            ub=desired_speed ** 2,
            soft_constraint=True,
            maximum_violation=max_speed ** 2,
            penalty_term_cons=1e-3
        )

        # constraint for collisions
        # for a, b in combinations(list(range(self.visible_agents)), 2):
        #     rad_sum = self.radius[a] + self.radius[b]
        #     mpc.set_nl_cons(
        #         f'collision_avoidance_constraint{a}/{b}',
        #         -casadi.sum2((m.x[f'pos{a}'] - m.x[f'pos{b}']) ** 2),
        #         ub=-(rad_sum ** 2),
        #         # soft_constraint=True,
        #         # maximum_violation=(rad_sum / 2.) ** 2,
        #         # penalty_term_cons=1e-1
        #      )

        mpc.set_tvp_fun(self.update_time_varying_parameters)

        mpc.setup()
        return mpc

    def create_simulator(self):
        simulator = self.simulator
        params = {
            't_step': self.time_step
        }
        simulator.set_param(
            **params
        )

        simulator.set_tvp_fun(self.get_simulator_tvp)
        simulator.setup()
        return simulator

    def init_control_loop(self, pos):
        x0 = self.controller.x0
        x0['self_pos'] = pos

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

    def set_agent_states(
            self,
            positions: np.ndarray,
            velocities: np.ndarray
    ):
        self.agents_positions = positions
        self.agents_velocities = velocities

    def update_time_varying_parameters(self, time: float):
        template = self.controller.get_tvp_template()

        # TODO magic number!
        # controller -> params -> n_horizon + 1
        for i in range(20 + 1):
            for agent_idx in range(self.visible_agents):
                template['_tvp', i, f'pos{agent_idx}'] = self.agents_positions[agent_idx]
                template['_tvp', i, f'vel{agent_idx}'] = self.agents_velocities[agent_idx]
        return template

    def get_simulator_tvp(self, time: float):
        template = self.simulator.get_tvp_template()
        print(template.labels())
        for agent_idx in range(self.visible_agents):
            template[f'pos{agent_idx}'] = self.agents_positions[agent_idx]
            template[f'vel{agent_idx}'] = self.agents_velocities[agent_idx]
        return template


if __name__ == '__main__':

    mpc_model = ModelPredictiveControl(
        visible_agents=4,
        time_step=0.1,
        target=np.array([0, 0]),  # np.array([[0, 1], [150, 6], [10, 100], [150, 150]]),
        radius=np.full(shape=(4,), fill_value=2.),
        max_speed=50.,
        desired_speed=40.
    )
    position = np.array([[200, 250], [100, 100], [200, 100], [0, 0]])
    mpc_model.init_control_loop(pos=position[0])

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
    plt.savefig('data.png')
