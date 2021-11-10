import numpy as np
import pygame as pg


class ReciprocalVelocityObstacle:

    def __init__(self, agents_num):
        self.left_boundaries = np.empty([agents_num, agents_num - 1, 2])
        self.right_boundaries = np.empty([agents_num, agents_num - 1, 2])

        self.centers = np.empty([agents_num, agents_num - 1, 2])

    def compute_vo(
            self,
            positions: np.ndarray,
            velocities: np.ndarray,
            radiuses: np.ndarray
    ):
        idx = 0
        while idx < radiuses.size:
            pos = positions[idx]

            vels = np.delete(velocities, idx, axis=0)
            rads = np.delete(radiuses, idx, axis=0) + radiuses[idx]
            points = np.delete(positions, idx, axis=0)

            self.centers[idx] = pos + vels
            ref_points = pos - points

            # for rad, rp in zip(rads, ref_points):
            sqr_dist = np.square(np.linalg.norm(ref_points))
            sqr_rad = np.square(rads)

            base = sqr_rad / sqr_dist * ref_points
            orto_vec = np.concatenate((-ref_points[:, [1]], ref_points[:, [0]]), axis=1)
            bias = rads / sqr_dist * np.sqrt(sqr_dist - sqr_rad) * orto_vec

            self.left_boundaries[idx] = 10 * (base - bias - ref_points)
            self.right_boundaries[idx] = 10 * (base + bias - ref_points)
            idx += 1

    def draw_debug(self, win, agent_idx):
        for i, pos in enumerate(self.centers[agent_idx]):
            pos = np.array([int(v) for v in pos])
            # pg.draw.circle(win, (255, 255, 255), [int(c) for c in pos], 1)
            # try:
            pg.draw.polygon(
                win,
                (200, 200, 200, 1),
                [pos,
                 pos + self.left_boundaries[agent_idx][i],
                 pos + self.right_boundaries[agent_idx][i]]
            )
            # except TypeError:
            #     # print([pos,
            #     #        pos + self.left_boundaries[agent_idx][i],
            #     #        pos + self.right_boundaries[agent_idx][i]]
            #     #       )
            #     pass

    def compute_velocities(self, agents: list):
        pass


if __name__ == '__main__':
    rvo = ReciprocalVelocityObstacle()
