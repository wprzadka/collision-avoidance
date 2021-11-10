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
        self.radiuses = radiuses.reshape((-1, 1))
        self.max_speeds = max_speeds.reshape((-1, 1))
        self.desired_speeds = desired_speeds.reshape((-1, 1))
        self.velocities = np.zeros((agents_num, 2))
        self.targets = self.positions.copy()
        self.debug_agent = 0

    def move(self, delta_time: float):
        self.velocities = self.targets - self.positions
        norms = np.linalg.norm(self.velocities, axis=1).reshape((-1, 1))
        norms[norms < 1] = 1.
        self.velocities = self.velocities / norms * self.desired_speeds
        self.positions = self.positions + self.velocities * delta_time

    def set_targets(self, targets: np.ndarray):
        self.targets = targets


if __name__ == '__main__':
    a = Agents(
        agents_num=3,
        positions=np.zeros((3, 2)),
        radiuses=np.array([5, 7, 20]),
        max_speeds=np.array([5, 6, 2]),
        desired_speeds=np.array([2, 3, 1])
    )
    a.set_targets(np.array([[10, 5], [20, 3], [5, 20]]))
    while np.sum(a.targets - a.positions) > 1e-2:
        a.move(0.1)
        print(a.positions)
