import pygame as pg
import numpy as np

from environment.simulation import Simulation
from environment.agents import Agents
from VO.ReciprocalVelocityObstacle import ReciprocalVelocityObstacle
from ORCA.OptimalReciprocalColisionAvoidance import ORCA

if __name__ == '__main__':

    win_size = np.array([1200, 900])
    window = pg.display.set_mode(win_size)
    pg.display.set_caption("Collision Avoidance")

    sim = Simulation()
    # sim.random_initialize(scene_size=win_size, agents_num=10)

    agents_num = 20
    positions = np.empty((agents_num, 2))
    targets = np.empty((agents_num, 2))
    for i in range(agents_num):
        x = i * 2. * np.pi / agents_num
        positions[i] = 250. * np.array([np.cos(x), np.sin(x)])
        targets[i] = -positions[i]
        positions[i] += win_size / 2
        targets[i] += win_size / 2

    agents = Agents(
        agents_num=agents_num,
        positions=positions,
        radiuses=np.full((agents_num, 1), 10),
        max_speeds=np.full((agents_num, 1), 40.),
        desired_speeds=np.full((agents_num, 1), 30.),
        velocity_diff_range=np.full((agents_num, 1), 10.)
    )
    sim.initialize(agents, targets)
    model = ReciprocalVelocityObstacle(
        agents_num=agents_num,
        visible_agents_num=3,
        reciprocal=True,
        shoots_num=100
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

        new_velocities = model.compute_velocities(agents)
        model.draw_debug(window, agent_idx=agents.debug_agent)

        # print(agents.get_preferred_velocities() - new_velocities)
        agents.set_velocity(new_velocities)
        agents.move(delta_time)
        sim.update(window)

        pg.display.update()
