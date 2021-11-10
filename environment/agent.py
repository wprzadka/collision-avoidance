import numpy as np


class Agent:

    def __init__(
            self,
            pos: np.ndarray,
            rad: float,
            max_speed: float,
            desired_speed: float = None
    ):
        self.position = pos
        self.radius = rad
        self.max_speed = max_speed
        self.desired_speed = max_speed if not desired_speed else desired_speed
        self.velocity = np.zeros(2)
        self.target = pos

    def move(self, delta_time: float):
        self.velocity = self.target - self.position
        self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.desired_speed
        self.position = self.position + self.velocity * delta_time

    def set_target(self, target: np.ndarray):
        self.target = target


if __name__ == '__main__':
    a = Agent(np.zeros(2), 5, 20)
    a.set_target(np.array([10, 5]))
    while np.sum(a.target - a.position) > 1e-2:
        a.move(0.1)
        print(a.position)
