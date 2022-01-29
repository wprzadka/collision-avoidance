import numpy as np
import pygame as pg
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from environment.agents import Agents
from utilities.math_utils import det2d


class ReciprocalVelocityObstacle:

    def __init__(
            self,
            agents_num: int,
            visible_agents_num: int = None,
            reciprocal: bool = False,
            shoots_num: int = 200
    ):
        self.agents_num = agents_num
        self.visible_agents_num = visible_agents_num or (agents_num - 1)
        assert self.visible_agents_num < self.agents_num

        self.reciprocal = reciprocal
        self.shoots_num = shoots_num

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
            agents: Agents
    ):
        positions = agents.positions
        velocities = agents.velocities
        preferred_velocities = agents.get_preferred_velocities()
        max_speeds = agents.max_speeds
        velocity_diff_range = agents.velocity_diff_range
        radii = agents.radiuses
        nearest_neighbours = agents.get_nearest_neighbours(self.visible_agents_num)

        self.compute_vo(positions, velocities, radii, nearest_neighbours)

        new_velocities = np.empty_like(velocities)
        for idx, vel in enumerate(velocities):
            points = self.generate_random_points()
            accessible_vels = vel + points * velocity_diff_range[idx]

            speeds = np.linalg.norm(accessible_vels, axis=1)
            over_speed_idxs = speeds > max_speeds[idx]
            accessible_vels[over_speed_idxs] *= max_speeds[idx] / speeds[over_speed_idxs].reshape((-1, 1))
            penalties = self.calculate_penalties(
                positions=positions,
                velocities=velocities,
                preferred_velocity=preferred_velocities[idx],
                accessible_velocities=accessible_vels,
                radiuses=radii,
                nearest=nearest_neighbours,
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
            nearest: np.ndarray,
            agent_idx: int
    ):
        velocity_cost = np.linalg.norm(preferred_velocity - accessible_velocities, axis=1)
        time_to_collision = np.array([
            self.compute_time_to_collision(
                positions=positions,
                relative_vels=(2 * vel - velocities[agent_idx] if self.reciprocal else vel)
                              - velocities[nearest[agent_idx]],
                rads=radiuses,
                nearest=nearest[agent_idx],
                agent_idx=agent_idx
            ) for vel in accessible_velocities])
        # multiply by aggressiveness parameter weight
        # for a, b in zip(velocity_cost, 20 / time_to_collision):
        #     print(a, b)
        return velocity_cost + 40 / time_to_collision

    def compute_time_to_collision(
            self,
            positions: np.ndarray,
            relative_vels: np.ndarray,
            rads: np.ndarray,
            nearest: np.ndarray,
            agent_idx: int
    ) -> float:
        pos_diff = positions[nearest] - positions[agent_idx]
        rad_sum = (rads[agent_idx] + rads[nearest]).flatten()
        rad_sum_sq = rad_sum ** 2

        c_times = np.empty(nearest.size)

        det = np.array([det2d(v, p) for v, p in zip(relative_vels, pos_diff)])
        vel_dot = np.sum(relative_vels ** 2, axis=1)
        # discriminant of quadratic equation which lower solutions us time of collision with other agent
        # and upper solution is time when collision finishes
        discriminant = -(det ** 2) + rad_sum_sq * vel_dot
        for i, d_val in enumerate(discriminant):
            if d_val > 0:
                c_times[i] = (np.dot(relative_vels[i], pos_diff[i]) - np.sqrt(d_val)) / vel_dot[i]
            if d_val < 0 or c_times[i] < 0:
                c_times[i] = float("inf")
        return np.min(c_times)

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

    def generate_random_points(self):
        rads = np.sqrt(np.random.rand(self.shoots_num))
        alphas = np.random.rand(self.shoots_num) * 2 * np.pi

        xs = np.cos(alphas) * rads
        ys = np.sin(alphas) * rads

        # plt.figure(figsize=(8, 8))
        # plt.scatter(xs, ys)
        # plt.grid(True)
        # plt.savefig('shoots.png')
        return np.concatenate((xs.reshape((-1, 1)), ys.reshape((-1, 1))), axis=1)
