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
    models = []
    for i in range(agents_num):
        model = ModelPredictiveControl(
            visible_agents=agents_num-1,
            time_step=delta_time,
            target=agents.targets[i],
            radius=agents.radiuses,
            max_speed=agents.max_speeds[i],
            desired_speed=agents.desired_speeds[i]
        )
        model.init_control_loop(agents.positions[i])
        model.set_agent_states(
            positions=np.delete(agents.positions, i, axis=0),
            velocities=np.delete(agents.velocities, i, axis=0)
        )
        models.append(model)

    sim.start()
    clock = pg.time.Clock()

    paused = True

    keys = pg.key.get_pressed()
    if keys[pg.K_SPACE]:
        paused = False

    while sim.running:
        clock.tick(60)
        window.fill((60, 60, 60))

        for event in pg.event.get():
            if event.type == pg.QUIT:
                sim.running = False
        keys = pg.key.get_pressed()
        if keys[pg.K_SPACE]:
            paused = False

        if not paused:
            new_velocities = np.empty_like(agents.velocities)
            for i, model in enumerate(models):
                model.set_agent_states(
                    positions=np.delete(agents.positions, i, axis=0),
                    velocities=np.delete(agents.velocities, i, axis=0)
                )
                model.update_control_loop()
                new_velocities[i] = model.simulator.data['_u'][-1]

            agents.set_velocity(new_velocities)
            agents.move(delta_time)

            for i, model in enumerate(models):
                model.state = agents.positions[i]

        sim.update(window)
        pg.display.update()

    rcParams['axes.grid'] = True
    rcParams['font.size'] = 18

    for i, model in enumerate(models):
        fig, ax, graphics = do_mpc.graphics.default_plot(model.controller.data, figsize=(16, 9))
        graphics.plot_results()
        graphics.reset_axes()
        plt.savefig(f'small_MPC_plots{i}.png')
