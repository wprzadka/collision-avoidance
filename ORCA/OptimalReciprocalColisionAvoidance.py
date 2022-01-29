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
            time_step: float = 0.25,
            time_horizon: float = 10.
    ):
        self.agents_num = agents_num
        self.visible_agents_num = visible_agents_num or self.agents_num - 1
        self.time_horizon = time_horizon
        self.inv_time_horizon = 1 / self.time_horizon
        self.time_step = time_step
        self.inv_time_step = 1 / self.time_step

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

        for idx in range(positions.shape[0]):
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
                    vel_diff = rel_vel[oth] - rel_pos[oth] * self.inv_time_horizon
                    vel_diff_mag = np.linalg.norm(vel_diff)
                    # side relative to cone direction
                    side_dir = np.dot(vel_diff, rel_pos[oth])

                    if side_dir < 0 and side_dir ** 2 > rad_sum_sq[oth] * vel_diff_mag:
                        # projection on cut-off circle

                        vel_diff_normalized = vel_diff / vel_diff_mag

                        u[idx, oth] = vel_diff_normalized * (rad_sum[oth] * self.inv_time_horizon - vel_diff_mag)
                        # right parallel vector to vel_diff (v.y, -v.x)
                        self.line_dir[idx, oth] = np.array([vel_diff_normalized[1], -vel_diff_normalized[0]])
                    else:
                        # projection on cone edges

                        det = self.det2d(rel_pos[oth], vel_diff)
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
                    vel_diff = rel_vel[oth] - rel_pos[oth] * self.inv_time_step
                    vel_diff_mag = np.linalg.norm(vel_diff)
                    vel_diff_normalized = vel_diff / vel_diff_mag

                    self.line_dir[idx, oth] = np.array([vel_diff_normalized[1], -vel_diff_normalized[0]])
                    u[idx, oth] = (rad_sum[oth] * self.inv_time_step - vel_diff_mag) * vel_diff_normalized

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
            temp_vel, failed_at = self.linear_prog_2d(
                agents.max_speeds[idx],
                pref_vel[idx],
                idx
            )

            if failed_at == self.line_point.shape[1]:
                new_vel[idx] = temp_vel
            else:
                # linear programming in 2D failed
                new_vel[idx] = self.linear_prog_3d(
                    agents.max_speeds[idx],
                    idx,
                    failed_at,
                    temp_vel
                )

        # for i in range(self.agents_num):
            # assert np.linalg.norm(agents.velocities[i]) < agents.max_speeds[i] + 0.00001
        return new_vel

    def linear_prog_3d(
            self,
            max_speed: float,
            agent_idx: int,
            line_nr: int,
            curr_velocity: np.ndarray
    ) -> np.ndarray:

        line_point = self.line_point[agent_idx]
        line_dir = self.line_dir[agent_idx]

        distance = 0.0
        for curr_line in range(line_nr, line_point.shape[0]):
            vel_displacement = line_point[curr_line] - curr_velocity
            det = self.det2d(line_dir[curr_line], vel_displacement)
            if det < distance:
                # velocity satisfies current line constraint
                continue
            # weaken the constraints due to make it able to satisfy
            temp_line_point = np.empty((curr_line, 2))
            temp_line_dir = np.empty((curr_line, 2))

            for j in range(0, curr_line):
                det = self.det2d(line_dir[curr_line], line_dir[j])
                if np.abs(det) < np.finfo(float).eps:
                    # lines are parallel
                    if np.dot(line_dir[curr_line], line_dir[j]) > 0:
                        # directions are this same
                        continue
                    else:
                        temp_line_point[j] = 0.5 * (line_point[j] + line_point[curr_line])
                else:
                    temp_line_point[j] = line_point[curr_line] + \
                                         line_dir[curr_line] * \
                                         (self.det2d(line_dir[j], line_point[curr_line] - line_point[j]) / det)

                dir_diff = line_dir[j] - line_dir[curr_line]
                temp_line_dir[j] = dir_diff / np.linalg.norm(dir_diff)

            # self.line_dir[agent_idx, :curr_line] = temp_line_dir
            # self.line_point[agent_idx, :curr_line] = temp_line_point

            pref_velocity = np.array([-line_dir[curr_line, 1], line_dir[curr_line, 0]])
            curr_velocity, _ = self.linear_prog_2d(
                max_speed,
                pref_velocity,
                agent_idx,
                temp_line_point,
                temp_line_dir,
                optimize_dir=True
            )
            distance = self.det2d(line_dir[curr_line], line_point[curr_line] - curr_velocity)
        return curr_velocity

    def linear_prog_2d(
            self,
            max_speed: float,
            pref_vel: np.ndarray,
            agent_idx: int,
            line_point: np.ndarray = None,
            line_dir: np.ndarray = None,
            optimize_dir: bool = False
    ) -> Tuple[np.ndarray, int]:
        if not optimize_dir:
            best_vel = pref_vel
        else:
            pref_vel_mag = np.linalg.norm(pref_vel)
            best_vel = pref_vel / pref_vel_mag * max_speed

        if line_point is None:
            line_point = self.line_point[agent_idx]
        if line_dir is None:
            line_dir = self.line_dir[agent_idx]

        for oth in range(line_point.shape[0]):
            vel_displacement = line_point[oth] - best_vel
            det = self.det2d(line_dir[oth], vel_displacement)
            if det > 0:
                # constraint is violated
                new_vel, succeed = self.linear_prog_1d(max_speed, pref_vel, agent_idx, line_nr=oth, optimize_dir=optimize_dir)
                if succeed:
                    best_vel = new_vel
                else:
                    return best_vel, oth
        # assert np.linalg.norm(best_vel) < max_speed + 0.00001
        return best_vel, self.visible_agents_num

    def linear_prog_1d(
            self,
            max_speed: float,
            pref_vel: np.ndarray,
            agent_idx: int,
            line_nr: int,
            line_point: np.ndarray = None,
            line_dir: np.ndarray = None,
            optimize_dir: bool = False
    ) -> Tuple[np.ndarray, bool]:

        if line_dir is None:
            line_dir = self.line_dir[agent_idx]
        if line_point is None:
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

            denominator = self.det2d(line_dir[line_nr], line_dir[oth])
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

        if not optimize_dir:
            t = np.dot(line_dir[line_nr], pref_vel - line_point[line_nr])

            # cut t in constraints
            t = max(t, left)
            t = min(t, right)

            # assert np.linalg.norm(line_point[line_nr] + t * line_dir[line_nr]) < max_speed + 0.00001
            return line_point[line_nr] + t * line_dir[line_nr], True
        else:
            # optimize direction
            if np.dot(pref_vel, line_dir[line_nr]) > 0.:
                return line_point[line_nr] + right * line_dir[line_nr], True
            else:
                return line_point[line_nr] + left * line_dir[line_nr], True

    @staticmethod
    def det2d(fst: np.ndarray, snd: np.ndarray) -> float:
        return fst[0] * snd[1] - fst[1] * snd[0]

    def draw_debug(self, win: pg.Surface, agent_idx: int):

        # width = win.get_width()
        for point, direction in zip(self.line_point_translated[agent_idx], self.line_dir[agent_idx]):
            # beg = point + ((0 - point[0]) / direction[0]) * direction
            # end = point + ((width - point[0]) / direction[0]) * direction
            beg = point - 1000 * direction
            end = point + 1000 * direction
            pg.draw.line(win, (200., 200., 200.), beg, end)
