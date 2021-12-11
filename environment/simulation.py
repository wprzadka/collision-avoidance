import numpy as np
import pygame as pg
from agents import Agents
from VO.ReciprocalVelocityObstacle import ReciprocalVelocityObstacle


class Simulation:

    def __init__(self):
        self.agents = None
        self.running = False
        self.last_update_time = 0

    def initialize(
            self,
            agents: Agents,
            targets: np.ndarray
    ):
        self.agents = agents
        agents.set_targets(targets)

    def start(self):
        self.running = True
        self.last_update_time = pg.time

    @staticmethod
    def get_random_points(points_num: int, width: int, height: int) -> np.ndarray:
        x_positions = np.random.normal(width / 2, width / 8, (points_num, 1))
        y_positions = np.random.normal(height / 2, height / 8, (points_num, 1))
        return np.concatenate((x_positions, y_positions), axis=1)

    def update(self, win: pg.display):
        self.last_update_time = pg.time
        for i, (pos, rad) in enumerate(zip(self.agents.positions, self.agents.radiuses)):
            color = (20, 255, 20) if i == agents.debug_agent else (20, 145, 220)
            pg.draw.circle(
                win,
                color,
                [int(v) for v in pos],
                int(rad)
            )
            pg.draw.circle(
                win,
                (100, 150, 255),
                [int(v) for v in pos],
                int(rad),
                1
            )
        for target in self.agents.targets:
            pg.draw.circle(
                win,
                (220, 70, 70),
                [int(v) for v in target],
                1
            )
        pg.draw.line(
            win,
            (150, 255, 150),
            agents.positions[agents.debug_agent],
            agents.positions[agents.debug_agent] + agents.velocities[agents.debug_agent]
        )
        pg.draw.line(
            win,
            (150, 150, 255),
            agents.positions[agents.debug_agent],
            agents.positions[agents.debug_agent] + agents.preferred_velocities[agents.debug_agent]
        )


if __name__ == '__main__':

    win_size = (1200, 900)
    window = pg.display.set_mode(win_size)
    pg.display.set_caption("Collision Avoidance")

    sim = Simulation()
    # sim.random_initialize(scene_size=win_size, agents_num=10)

    agents = Agents(
        agents_num=4,
        positions=np.array([
            [100., 200.],
            [400., 200.],
            [200., 300.],
            [400., 500.]
        ]),
        radiuses=np.full((4, 1), 10),
        max_speeds=np.full((4, 1), 100.),
        desired_speeds=np.full((4, 1), 75.),
        velocity_diff_range=np.full((4, 1), 10.)
    )
    targets = np.array([
        [400., 500.],
        [200., 500.],
        [500., 400.],
        [100., 200.]
    ])
    sim.initialize(agents, targets)
    rvo = ReciprocalVelocityObstacle(agents_num=agents.agents_num, reciprocal=True)

    sim.start()
    clock = pg.time.Clock()

    delta_time = 1 / 60.
    while sim.running:
        clock.tick(60)
        window.fill((60, 60, 60))

        for event in pg.event.get():
            if event.type == pg.QUIT:
                sim.running = False

        rvo.compute_vo(agents.positions, agents.velocities, agents.radiuses)
        new_velocities = rvo.compute_velocities(
            agents.positions,
            agents.velocities,
            agents.get_preferred_velocities(),
            agents.max_speeds,
            agents.velocity_diff_range,
            agents.radiuses,
            200
        )
        rvo.draw_debug(window, agent_idx=agents.debug_agent)

        # print(agents.get_preferred_velocities() - new_velocities)
        agents.set_velocity(new_velocities)
        agents.move(delta_time)
        sim.update(window)

        pg.display.update()
