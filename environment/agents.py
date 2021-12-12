import numpy as np


class Agents:

    def __init__(
            self,
            agents_num: int,
            positions: np.ndarray,
            radiuses: np.ndarray,
            max_speeds: np.ndarray,
            desired_speeds: np.ndarray,
            velocity_diff_range: np.ndarray
    ):
        self.agents_num = agents_num
        self.positions = positions
        self.radiuses = radiuses.reshape((-1, 1))
        self.max_speeds = max_speeds.reshape((-1, 1))
        self.desired_speeds = desired_speeds.reshape((-1, 1))
        self.velocity_diff_range = velocity_diff_range.reshape((-1, 1))
        self.velocities = np.zeros((agents_num, 2))
        self.preferred_velocities = np.zeros((agents_num, 2))
        self.targets = self.positions.copy()
        self.debug_agent = 0

    def move(self, delta_time: float):
        self.positions = self.positions + self.velocities * delta_time

    def set_velocity(self, new_velocities: np.ndarray):
        self.velocities = new_velocities

    def get_preferred_velocities(self):
        self.preferred_velocities = self.targets - self.positions
        norms = np.linalg.norm(self.preferred_velocities, axis=1).reshape((-1, 1))

        condition = norms > 1.

        multipliers = np.empty_like(self.desired_speeds)
        multipliers[~condition] = 0.
        multipliers[condition] = self.desired_speeds[condition] / norms[condition]

        self.preferred_velocities = self.preferred_velocities * multipliers

        return self.preferred_velocities

    def set_targets(self, targets: np.ndarray):
        self.targets = targets


if __name__ == '__main__':
    a = Agents(
        agents_num=3,
        positions=np.zeros((3, 2)),
        radiuses=np.array([5, 7, 20]),
        max_speeds=np.array([5, 6, 2]),
        desired_speeds=np.array([2, 3, 1]),
        velocity_diff_range=np.array([2, 2, 2])
    )
    a.set_targets(np.array([[10, 5], [20, 3], [5, 20]]))
    while np.sum(a.targets - a.positions) > 1e-2:
        a.move(0.1)
        print(a.positions)
