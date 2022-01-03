import numpy as np
import pygame as pg
from environment.agents import Agents
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
            color = (20, 255, 20) if i == self.agents.debug_agent else (20, 145, 220)
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
        for i, target in enumerate(self.agents.targets):
            pg.draw.circle(
                win,
                (220, 70, 70),
                [int(v) for v in target],
                4 if i == self.agents.debug_agent else 1
            )

        pg.draw.line(
            win,
            (150, 255, 150),
            self.agents.positions[self.agents.debug_agent],
            self.agents.positions[self.agents.debug_agent]
            + self.agents.velocities[self.agents.debug_agent]
        )
        pg.draw.line(
            win,
            (150, 150, 255),
            self.agents.positions[self.agents.debug_agent],
            self.agents.positions[self.agents.debug_agent]
            + self.agents.preferred_velocities[self.agents.debug_agent]
        )

        nearest = self.agents.get_nearest_neighbours(1)
        for idx, (pos, neig) in enumerate(list(zip(self.agents.positions, nearest))):
            if self.is_colliding(idx, neig[0]):
                pg.draw.circle(
                    win,
                    (255, 0, 0),
                    [int(v) for v in pos],
                    int(self.agents.radiuses[idx]),
                    1
                )

    def is_colliding(self, fst: int, snd: int) -> bool:
        return np.linalg.norm(self.agents.positions[fst] - self.agents.positions[snd]) \
               < self.agents.radiuses[fst] + self.agents.radiuses[snd]


if __name__ == '__main__':

    win_size = (1200, 900)
    window = pg.display.set_mode(win_size)
    pg.display.set_caption("Collision Avoidance")

    sim = Simulation()
    # sim.random_initialize(scene_size=win_size, agents_num=10)

    agents_num = 6
    agents = Agents(
        agents_num=agents_num,
        positions=np.array([
            [100., 200.],
            [400., 200.],
            [200., 300.],
            [300., 300.],
            [400., 300.],
            [400., 500.]
        ]),
        radiuses=np.full((agents_num, 1), 10),
        max_speeds=np.full((agents_num, 1), 100.),
        desired_speeds=np.full((agents_num, 1), 75.),
        velocity_diff_range=np.full((agents_num, 1), 20.)
    )
    targets = np.array([
        [400., 500.],
        [200., 500.],
        [500., 400.],
        [100., 250.],
        [700., 260.],
        [140., 200.]
    ])
    sim.initialize(agents, targets)
    rvo = ReciprocalVelocityObstacle(visible_agents_num=agents.agents_num)  # , reciprocal=True)

    sim.start()
    clock = pg.time.Clock()

    delta_time = 1 / 60.
    while sim.running:
        clock.tick(60)
        window.fill((60, 60, 60))

        for event in pg.event.get():
            if event.type == pg.QUIT:
                sim.running = False

        new_velocities = rvo.compute_velocities(
            agents.positions,
            agents.velocities,
            agents.get_preferred_velocities(),
            agents.max_speeds,
            agents.velocity_diff_range,
            agents.radiuses,
            shoots_num=200
        )
        rvo.draw_debug(window, agent_idx=agents.debug_agent)

        # print(agents.get_preferred_velocities() - new_velocities)
        agents.set_velocity(new_velocities)
        agents.move(delta_time)
        sim.update(window)

        pg.display.update()
