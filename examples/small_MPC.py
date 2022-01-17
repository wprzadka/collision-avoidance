import casadi
import pygame as pg
import numpy as np

from environment.simulation import Simulation
from environment.agents import Agents
from MPC.model_predictive_control import ModelPredictiveControl

from matplotlib import rcParams
import matplotlib.pyplot as plt
import do_mpc


if __name__ == '__main__':

    win_size = (1200, 900)
    window = pg.display.set_mode(win_size)
    pg.display.set_caption("Collision Avoidance")

    sim = Simulation()
    # sim.random_initialize(scene_size=win_size, agents_num=10)

    agents_num = 3
    agents = Agents(
        agents_num=agents_num,
        positions=np.array([
            [600., 485.],
            [100., 500.],
            [600., 515.]
        ]),
        radiuses=np.full((agents_num, 1), 10),
        max_speeds=np.full((agents_num, 1), 100.),
        desired_speeds=np.full((agents_num, 1), 75.),
        velocity_diff_range=np.full((agents_num, 1), 20.)
    )
    targets = np.array([
        [100., 515.],
        [600., 500.],
        [100., 485.]
    ])
    sim.initialize(agents, targets)

    delta_time = 1 / 60.
    model = ModelPredictiveControl(
        visible_agents=agents_num,
        time_step=delta_time,
        own_target=agents.targets,
        max_speed=agents.max_speeds[0],
        desired_speed=agents.desired_speeds[0]
    )
    model.init_control_loop(agents.positions)

    sim.start()
    clock = pg.time.Clock()

    while sim.running:
        clock.tick(60)
        window.fill((60, 60, 60))

        for event in pg.event.get():
            if event.type == pg.QUIT:
                sim.running = False

        model.update_control_loop()
        # new_velocities = agents.get_preferred_velocities()

        # if casadi.norm_2(model.state - model.own_target) > 0.1:
        # print(model.simulator.data['_u'][-1].reshape(-1, 2))
        new_velocities = model.simulator.data['_u'][-1].reshape(-1, 2)
        # else:
            # sim.running = False
            # new_velocities[0] = np.zeros(2)

        # new_velocities = model.compute_velocities(agents)
        # model.draw_debug(window, agent_idx=agents.debug_agent)


        agents.set_velocity(new_velocities)
        agents.move(delta_time)
        model.state = agents.positions.reshape(-1, 1)

        sim.update(window)

        pg.display.update()

    rcParams['axes.grid'] = True
    rcParams['font.size'] = 18

    fig, ax, graphics = do_mpc.graphics.default_plot(model.controller.data, figsize=(16, 9))
    graphics.plot_results()
    graphics.reset_axes()
    plt.savefig('data.png')
