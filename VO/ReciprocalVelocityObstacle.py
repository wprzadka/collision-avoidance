import numpy as np
import pygame as pg
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


class ReciprocalVelocityObstacle:

    def __init__(
            self,
            agents_num: int,
            visible_agents_num: int = None,
            reciprocal: bool = False
    ):
        self.agents_num = agents_num
        self.visible_agents_num = visible_agents_num or (agents_num - 1)
        assert self.visible_agents_num < self.agents_num

        self.reciprocal = reciprocal
        self.centers = np.empty([self.agents_num, self.visible_agents_num, 2])
        self.left_boundaries = np.empty([self.agents_num, self.visible_agents_num, 2])
        self.right_boundaries = np.empty([self.agents_num, self.visible_agents_num, 2])
        self.left_normals = None
        self.right_normals = None

    def compute_vo(
            self,
            positions: np.ndarray,
            velocities: np.ndarray,
            radiuses: np.ndarray,
            nearest_neighbours: np.ndarray
    ):
        idx = 0
        while idx < self.agents_num:
            agent_pos = positions[idx]
            agent_vel = velocities[idx]

            vels = velocities[nearest_neighbours[idx]]
            rads = radiuses[nearest_neighbours[idx]]
            points = positions[nearest_neighbours[idx]]

            self.centers[idx] = agent_pos + (vels if not self.reciprocal else (vels + agent_vel) / 2)
            ref_points = agent_pos - points

            sqr_dist = np.square(np.linalg.norm(ref_points))
            sqr_rad = np.square(rads)

            base = sqr_rad / sqr_dist * ref_points
            orto_vec = np.concatenate((-ref_points[:, [1]], ref_points[:, [0]]), axis=1)
            bias = rads / sqr_dist * np.sqrt(sqr_dist - sqr_rad) * orto_vec

            self.left_boundaries[idx] = base - bias - ref_points
            self.right_boundaries[idx] = base + bias - ref_points
            idx += 1
        self.compute_cone_normals()

    def compute_cone_normals(self):
        self.left_normals = np.concatenate(
            (-self.left_boundaries[:, :, 1], self.left_boundaries[:, :, 0]),
            axis=1
        )
        self.right_normals = np.concatenate(
            (self.left_boundaries[:, :, 1], -self.left_boundaries[:, :, 0]),
            axis=1
        )

    def draw_debug(self, win, agent_idx):
        for i, pos in enumerate(self.centers[agent_idx]):
            pos = np.array([int(v) for v in pos])
            pg.draw.polygon(
                win,
                (200, 200, 200, 1),
                [pos,
                 pos + 10 * self.left_boundaries[agent_idx][i],
                 pos + 10 * self.right_boundaries[agent_idx][i]]
            )

    def compute_velocities(
            self,
            positions: np.ndarray,
            velocities: np.ndarray,
            preferred_velocities: np.ndarray,
            max_speeds: np.ndarray,
            velocity_diff_range: np.ndarray,
            radiuses: np.ndarray,
            nearest_neighbours: np.ndarray,
            shoots_num: int
    ):
        self.compute_vo(positions, velocities, radiuses, nearest_neighbours)

        new_velocities = np.empty_like(velocities)
        for idx, vel in enumerate(velocities):
            points = self.generate_random_points(shoots_num)
            accessible_vels = vel + points * velocity_diff_range[idx]

            speeds = np.linalg.norm(accessible_vels, axis=1)
            over_speed_idxs = speeds > max_speeds[idx]
            accessible_vels[over_speed_idxs] *= max_speeds[idx] / speeds[over_speed_idxs].reshape((-1, 1))
            penalties = self.calculate_penalties(
                positions=positions,
                velocities=velocities,
                preferred_velocity=preferred_velocities[idx],
                accessible_velocities=accessible_vels,
                radiuses=radiuses,
                agent_idx=idx
            )
            new_velocities[idx] = accessible_vels[np.argmin(penalties)]

            # self.plot_debug_velocity(accessible_vels=accessible_vels)
        return new_velocities

    def calculate_penalties(
            self,
            positions: np.ndarray,
            velocities: np.ndarray,
            preferred_velocity: np.ndarray,
            accessible_velocities: np.ndarray,
            radiuses: np.ndarray,
            agent_idx: int
    ):
        velocity_cost = np.linalg.norm(preferred_velocity - accessible_velocities, axis=1)
        time_to_collision = np.array([
            self.compute_time_to_collision(
                new_velocity=vel,
                positions=positions,
                velocities=velocities,
                rads=radiuses,
                agent_idx=agent_idx
            ) for vel in accessible_velocities])
        # multiply by aggressiveness parameter weight
        # for a, b in zip(velocity_cost, 20 / time_to_collision):
        #     print(a, b)
        return velocity_cost + 40 / time_to_collision

    def compute_time_to_collision(
            self,
            new_velocity: np.ndarray,
            positions: np.ndarray,
            velocities: np.ndarray,
            rads: np.ndarray,
            agent_idx: int
    ) -> float:
        pos_diff = positions[agent_idx] - np.delete(positions, agent_idx, axis=0)
        vel_diff = new_velocity - np.delete(velocities, agent_idx, axis=0)
        c = np.sum(np.square(pos_diff), axis=1) - \
            np.square(rads[agent_idx] + np.delete(rads, agent_idx, axis=0)).flatten()
        b = 2 * np.sum(pos_diff * vel_diff, axis=1)
        a = np.sum(np.square(vel_diff), axis=1)

        delta = np.square(b) - 4 * a * c
        if all(delta <= 0):
            return np.infty
        times = (-b[delta > 0] - np.sqrt(delta[delta > 0])) / (2 * a[delta > 0])
        if all(times <= 0):
            return np.infty
        return min(times[times > 0])

    def plot_debug_velocity(self, accessible_vels: np.ndarray, penalties: np.ndarray):
        fig, ax = plt.subplots()

        fig.size = (8, 8)
        plt.xlim((np.min(accessible_vels[:, 0]) - 1, np.max(accessible_vels[:, 0]) + 1))
        plt.ylim((np.min(accessible_vels[:, 1]) - 1, np.max(accessible_vels[:, 1]) + 1))

        sc = ax.scatter(accessible_vels[:, 0], accessible_vels[:, 1], c=penalties)
        plt.colorbar(sc)

        for i, center in enumerate(self.centers[1:], 1):
            left = center + 10 * self.left_boundaries[i]
            right = center + 10 * self.right_boundaries[i]
            p = Polygon(np.concatenate((center, left, right)), closed=True, color=(1, 0, 0, 0.1))
            ax.add_patch(p)

        plt.grid(True)
        plt.savefig('shoots.png')

    def generate_random_points(
            self,
            shoots_num: int
    ):
        rads = np.sqrt(np.random.rand(shoots_num))
        alphas = np.random.rand(shoots_num) * 2 * np.pi

        xs = np.cos(alphas) * rads
        ys = np.sin(alphas) * rads

        # plt.figure(figsize=(8, 8))
        # plt.scatter(xs, ys)
        # plt.grid(True)
        # plt.savefig('shoots.png')
        return np.concatenate((xs.reshape((-1, 1)), ys.reshape((-1, 1))), axis=1)


if __name__ == '__main__':
    rvo = ReciprocalVelocityObstacle(2)
    # print(rvo.generate_random_points(1000))
    rvo.compute_vo(
        np.array([[0, 0], [1, 2]]),
        np.array([[5, 2], [4, 3]]),
        np.array([1, 1]),
        np.array([[1], [0]])
    )
    rvo.compute_velocities(
        np.array([[0, 0], [1, 2]]),
        np.array([[5, 2], [4, 3]]),
        np.array([[5, 5], [6, 4]]),
        np.array([7, 7]),
        np.array([2, 2]),
        np.array([1, 1]),
        np.array([[1], [0]]),
        1000
    )
