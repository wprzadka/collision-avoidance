import numpy as np
import pygame as pg
from agent import Agent


class Simulation:

    def __init__(self):
        self.agents = None
        self.running = False
        self.last_update_time = 0

    def initialize(
            self,
            agents: np.ndarray,
            targets: np.ndarray
    ):
        self.agents = agents
        for a, t in zip(agents, targets):
            a.set_target(t)

    def random_initialize(
            self,
            scene_size: tuple,
            agents_num: int,
            max_speed: float = 40.,
            radius: float = 8.,
    ):
        width, height = scene_size
        positions = self.get_random_points(agents_num, width, height)
        self.agents = [
            Agent(
                pos=pos,
                rad=radius,
                max_speed=max_speed
            )
            for pos in positions
        ]
        targets = self.get_random_points(agents_num, width, height)
        for a, t in zip(self.agents, targets):
            a.set_target(t)

    def start(self):
        self.running = True
        self.last_update_time = pg.time

    @staticmethod
    def get_random_points(points_num: int, width: int, height: int) -> np.ndarray:
        x_positions = np.random.normal(width / 2, width / 8, (points_num, 1))
        y_positions = np.random.normal(height / 2, height / 8, (points_num, 1))
        return np.concatenate((x_positions, y_positions), axis=1)

    def update(self, win: pg.display):
        dt = 1 / 60.
        self.last_update_time = pg.time
        for a in self.agents:
            a.move(dt)
            pg.draw.circle(
                win,
                (0, 125, 200),
                [int(v) for v in a.position],
                int(a.radius),
                1
            )
            pg.draw.circle(
                win,
                (200, 50, 50),
                [int(v) for v in a.target],
                1,
                1
            )


if __name__ == '__main__':

    win_size = (1200, 900)
    window = pg.display.set_mode(win_size)
    pg.display.set_caption("Collision Avoidance")

    sim = Simulation()
    # sim.random_initialize(scene_size=win_size, agents_num=10)

    agents = np.array([
        Agent(np.array([100, 200]), 20, 40),
        Agent(np.array([400, 200]), 20, 40)
    ])
    targets = np.array([
        [400, 200],
        [100, 200]
    ])
    sim.initialize(agents, targets)

    sim.start()
    clock = pg.time.Clock()

    step = 0
    steps_limit = 400
    while sim.running and step < steps_limit:
        clock.tick(60)
        window.fill((60, 60, 60))
        sim.update(window)
        step += 1
        pg.display.update()
