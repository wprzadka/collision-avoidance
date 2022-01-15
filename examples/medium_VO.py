import pygame as pg
import numpy as np

from environment.simulation import Simulation
from environment.agents import Agents
from VO.ReciprocalVelocityObstacle import ReciprocalVelocityObstacle


if __name__ == '__main__':

    win_size = (1200, 900)
    window = pg.display.set_mode(win_size)
    pg.display.set_caption("Collision Avoidance")

    sim = Simulation()
    # sim.random_initialize(scene_size=win_size, agents_num=10)

    agents_num = 10
    agents = Agents(
        agents_num=agents_num,
        positions=np.array([
            [100., 200.],
            [400., 600.],
            [200., 300.],
            [300., 300.],
            [400., 300.],
            [450., 350.],
            [230., 30.],
            [700., 350.],
            [420., 370.],
            [460., 510.]
        ]),
        radiuses=np.full((agents_num, 1), 10),
        max_speeds=np.full((agents_num, 1), 100.),
        desired_speeds=np.full((agents_num, 1), 75.),
        velocity_diff_range=np.full((agents_num, 1), 20.)
    )
    targets = np.array([
        [400., 500.],
        [100., 200.],
        [500., 400.],
        [100., 250.],
        [700., 260.],
        [180., 700.],
        [160., 600.],
        [120., 150.],
        [270., 50.],
        [200., 100.],
    ])
    sim.initialize(agents, targets)
    rvo = ReciprocalVelocityObstacle(
        agents_num=agents_num,
        visible_agents_num=3
        # reciprocal=True
    )

    sim.start()
    clock = pg.time.Clock()

    delta_time = 1 / 60.
    while sim.running:
        clock.tick(60)
        window.fill((60, 60, 60))

        for event in pg.event.get():
            if event.type == pg.QUIT:
                sim.running = False

        new_velocities = rvo.compute_velocities(agents, shoots_num=100)
        rvo.draw_debug(window, agent_idx=agents.debug_agent)

        # print(agents.get_preferred_velocities() - new_velocities)
        agents.set_velocity(new_velocities)
        agents.move(delta_time)
        sim.update(window)

        pg.display.update()
