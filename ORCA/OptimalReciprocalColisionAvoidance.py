import numpy as np
import pygame as pg
import pygame.draw
from typing import Tuple

from environment.agents import Agents


class ORCA:

    def __init__(
            self,
            agents_num: int,
            visible_agents_num: int = None,
            time_horizon: float = 1.
    ):
        self.agents_num = agents_num
        self.visible_agents_num = visible_agents_num or self.agents_num - 1
        self.time_horizon = time_horizon

        self.line_point = np.empty((self.agents_num, self.visible_agents_num, 2))
        self.line_point_translated = np.empty((self.agents_num, self.visible_agents_num, 2))
        self.line_dir = np.empty((self.agents_num, self.visible_agents_num, 2))

    def compute_lines(
            self,
            positions: np.ndarray,
            velocities: np.ndarray,
            radii: np.ndarray,
            nearest: np.ndarray
    ):
        for idx in range(len(positions)):
            rel_vel = velocities[idx] - velocities[nearest[idx]]
            rel_pos = positions[nearest[idx]] - positions[idx]
            dist_sq = np.sum(rel_pos ** 2, axis=1)
            rad_sum = radii[idx] + radii[nearest[idx]]
            rad_sum_sq = rad_sum ** 2

            u = np.empty((*nearest.shape, 2))

            for oth in range(len(nearest[idx])):
                if dist_sq[oth] > rad_sum_sq[oth]:
                    # no collision occurred

                    # difference between relative velocity and minimal velocity that leads to positions equality
                    vel_diff = rel_vel[oth] - rel_pos[oth] / self.time_horizon
                    vel_diff_mag = np.linalg.norm(vel_diff)
                    # side relative to cone direction
                    side_dir = np.dot(vel_diff, rel_pos[oth])

                    if side_dir < 0 and side_dir ** 2 > rad_sum_sq[oth] * vel_diff_mag:
                        # projection on cut-off circle

                        vel_diff_normalized = vel_diff / vel_diff_mag

                        u[idx, oth] = vel_diff_normalized * (rad_sum[oth] / self.time_horizon - vel_diff_mag)
                        # right parallel vector to vel_diff (v.y, -v.x)
                        self.line_dir[idx, oth] = np.array([vel_diff_normalized[1], -vel_diff_normalized[0]])
                    else:
                        # projection on cone edges

                        # det = np.linalg.det(rel_pos, vel_diff)
                        det = rel_pos[oth, 0] * vel_diff[1] - rel_pos[oth, 1] * vel_diff[0]
                        leg = np.sqrt(dist_sq[oth] - rad_sum_sq[oth])

                        if det > 0:
                            # right edge
                            self.line_dir[idx, oth] = np.hstack([
                                rel_pos[oth, 0] * leg - rel_pos[oth, 1] * rad_sum[oth],
                                rel_pos[oth, 0] * rad_sum[oth] + rel_pos[oth, 1] * leg,
                            ]) / dist_sq[oth]
                        else:
                            # left edge
                            self.line_dir[idx, oth] = -np.hstack([
                                rel_pos[oth, 0] * leg + rel_pos[oth, 1] * rad_sum[oth],
                                -rel_pos[oth, 0] * rad_sum[oth] + rel_pos[oth, 1] * leg,
                            ]) / dist_sq[oth]

                        dot = np.dot(rel_vel[oth], self.line_dir[idx, oth])
                        u[idx, oth] = dot * self.line_dir[idx][oth] - rel_vel[oth]
                else:
                    # collision occurred
                    pass

            # for idx in range(self.agents_num):
            for oth in range(self.visible_agents_num):
                self.line_point[idx, oth] = velocities[idx] + u[idx, oth] / 2.
                self.line_point_translated[idx, oth] = self.line_point[idx, oth] + positions[idx]

    def compute_velocities(self, agents: Agents) -> np.ndarray:
        positions = agents.positions
        velocities = agents.velocities
        radii = agents.radiuses
        nearest = agents.get_nearest_neighbours(self.visible_agents_num)

        self.compute_lines(positions, velocities, radii, nearest)

        pref_vel = agents.get_preferred_velocities()
        new_vel = np.empty_like(pref_vel)

        for idx in range(self.agents_num):
            temp_vel, fails_count = self.linear_prog_2d(
                agents.max_speeds[idx],
                pref_vel[idx],
                idx
            )

            if fails_count == 0:
                new_vel[idx] = temp_vel
            else:
                raise Exception("No solution")
                # linear programming 3D
                pass
        # for i in range(self.agents_num):
            # assert np.linalg.norm(agents.velocities[i]) < agents.max_speeds[i] + 0.00001
        return new_vel

    def linear_prog_2d(
            self,
            max_speed: float,
            pref_vel: np.ndarray,
            agent_idx: int
    ) -> Tuple[np.ndarray, int]:
        best_vel = pref_vel

        for oth in range(self.visible_agents_num):
            vel_displacement = self.line_point[agent_idx, oth] - best_vel
            det = self.line_dir[agent_idx, oth][0] * vel_displacement[1] - \
                  self.line_dir[agent_idx, oth][1] * vel_displacement[0]
            if det > 0:
                # constraint is violated
                new_vel, succeed = self.linear_prog_1d(max_speed, pref_vel, agent_idx, line_nr=oth)
                if succeed:
                    best_vel = new_vel
                else:
                    return best_vel, oth
        # assert np.linalg.norm(best_vel) < max_speed + 0.00001
        return best_vel, 0

    def linear_prog_1d(
            self,
            max_speed: float,
            pref_vel: np.ndarray,
            agent_idx: int,
            line_nr: int
    ) -> Tuple[np.ndarray, bool]:

        line_dir = self.line_dir[agent_idx]
        line_point = self.line_point[agent_idx]

        # check if intersection of velocities satisfying the max speed condition and ORCA line constraint is not null
        # with discriminant of quadratic equation that is looking for points of intersections between max speed circle
        # and ORCA constraint line
        coef_2b = np.dot(line_point[line_nr], line_dir[line_nr])
        discriminant = coef_2b ** 2 + \
                      max_speed ** 2 - \
                      np.dot(line_point[line_nr], line_point[line_nr])

        if discriminant < 0:
            return np.zeros(2), False

        # points of intersections
        # solutions of quadratic equatinos
        determinantSqrt = np.sqrt(discriminant)
        left = -coef_2b - determinantSqrt
        right = -coef_2b + determinantSqrt
        # a - term of equation equals 1

        for oth in range(line_nr):

            denominator = line_dir[line_nr, 0] * line_dir[oth, 1] - \
                          line_dir[line_nr, 1] * line_dir[oth, 0]
            if np.abs(denominator) < np.finfo(float).eps:
                if np.dot(line_dir[line_nr], line_dir[oth]) < 0:
                    return np.zeros(2), False
                continue

            point_diff = line_point[line_nr] - line_point[oth]
            numerator = line_dir[oth, 0] * point_diff[1] - \
                        line_dir[oth, 1] * point_diff[0]

            t = numerator / denominator

            if denominator > 0:
                right = min(t, right)
            else:
                left = max(t, left)

            if left > right:
                return np.zeros(2), False

        t = np.dot(line_dir[line_nr], pref_vel - line_point[line_nr])

        # cut t in constraints
        t = max(t, left)
        t = min(t, right)

        # assert np.linalg.norm(line_point[line_nr] + t * line_dir[line_nr]) < max_speed + 0.00001
        return line_point[line_nr] + t * line_dir[line_nr], True

    def draw_debug(self, win: pg.Surface, agent_idx: int):

        width = win.get_width()
        for point, direction in zip(self.line_point_translated[agent_idx], self.line_dir[agent_idx]):
            beg = point + ((0 - point[0]) / direction[0]) * direction
            end = point + ((width - point[0]) / direction[0]) * direction
            pg.draw.line(win, (200., 200., 200.), beg, end)
