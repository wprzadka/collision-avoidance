import numpy as np


class Agents:

    def __init__(
            self,
            agents_num: int,
            positions: np.ndarray,
            radiuses: np.ndarray,
            max_speeds: np.ndarray,
            desired_speeds: np.ndarray
    ):
        self.agents_num = agents_num
        self.positions = positions
        self.radiuses = radiuses
        self.max_speeds = max_speeds
        self.desired_speeds = desired_speeds
        self.velocities = np.zeros((2, agents_num))
        self.targets = self.positions.copy()

    def move(self, delta_time: float):
        self.velocities = self.targets - self.positions
        self.velocities = self.velocities / np.linalg.norm(self.velocities) * self.desired_speeds
        self.positions = self.positions + self.velocities * delta_time

    def set_targets(self, targets: np.ndarray):
        self.targets = targets


if __name__ == '__main__':
    a = Agents(
        agents_num=2,
        positions=np.zeros((2, 2)),
        radiuses=np.array([5, 7]),
        max_speeds=np.array([5, 6]),
        desired_speeds=np.array([2, 3])
    )
    a.set_targets(np.array([[10, 5], [20, 3]]))
    while np.sum(a.targets - a.positions) > 1e-2:
        a.move(0.1)
        print(a.positions)
