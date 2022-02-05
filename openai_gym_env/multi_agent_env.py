from abc import ABC

import numpy as np
import pygame as pg
from gym import spaces
import gym

from environment.simulation import Simulation
from environment.agents import Agents


class MultiAgentCollisionAvoidanceEnv(gym.Env, ABC):
    metadata = {'render.modes': ['human']}
    collision_penalty = 100.
    out_of_bounds_penalty = 10_000.
    distance_reward = 20.
    goal_reward = 500.

    def __init__(
            self,
            agents_num: int,
            visible_agents_num: int,
            max_speed: float,
            win_size: tuple = (1200, 900),
            distance_quantification: int = 5,
            time_step: float = 0.25,
            time_limit: int = 100
    ):
        super(MultiAgentCollisionAvoidanceEnv, self).__init__()

        self.max_agents_speed = max_speed
        self.agents_num = agents_num
        self.visible_agents_num = visible_agents_num

        self.curr_agent_action = 0
        self.next_step_velocities = np.empty(shape=(agents_num, 2))
        self.dones = np.full(shape=self.agents_num, fill_value=False, dtype=bool)
        self.rewards = np.zeros(shape=self.agents_num)

        self.win_size = win_size

        self.action_space = spaces.Box(
            shape=(2,),
            low=-1,
            high=1
        )

        self.observation_space = spaces.Dict({
            'velocity_x': spaces.Box(
                shape=(self.visible_agents_num,),
                low=-max_speed,
                high=max_speed
            ),
            'velocity_y': spaces.Box(
                shape=(self.visible_agents_num,),
                low=-max_speed,
                high=max_speed
            ),
            'position_x': spaces.Box(
                shape=(self.visible_agents_num,),
                low=0,
                high=self.win_size[0]
            ),
            'position_y': spaces.Box(
                shape=(self.visible_agents_num,),
                low=0,
                high=self.win_size[1]
            ),
            'target': spaces.Box(
                shape=(2,),
                low=np.zeros(2),
                high=np.array(self.win_size)
            ),
            'self_position': spaces.Box(
                shape=(2,),
                low=np.zeros(2),
                high=np.array(self.win_size)
            )
        })
        # simulation
        self.simulation = Simulation()
        self.initialize_simulation()
        self.time = 0
        self.time_step = time_step
        self.time_limit = time_limit
        self.closest_reached_distance = None
        self.distance_quantification = distance_quantification
        # rendering
        self.window = None
        # algorithm
        self.algorithm = None
        self.get_velocities = None

    def initialize_simulation(self):
        agents = Agents(
            agents_num=self.agents_num,
            positions=np.random.rand(self.agents_num, 2) * self.win_size,
            radiuses=np.full((self.agents_num, 1), 10),
            max_speeds=np.full((self.agents_num, 1), self.max_agents_speed),
            desired_speeds=np.full((self.agents_num, 1), 0.75 * self.max_agents_speed),
            velocity_diff_range=np.full((self.agents_num, 1), 10.)
        )
        targets = np.random.rand(self.agents_num, 2) * self.win_size
        self.simulation.initialize(agents, targets)

    def step(self, action: np.ndarray):

        if not self.dones[self.curr_agent_action]:
            self.next_step_velocities[self.curr_agent_action] = action * self.max_agents_speed

        # observation
        obs = self.get_observations(self.curr_agent_action)

        # current reward
        curr_reward = self.rewards[self.curr_agent_action]

        # done
        is_done = all(self.dones) or self.time > self.time_limit

        if self.curr_agent_action == self.agents_num - 1:
            self.time += self.time_step

            new_velocities = self.next_step_velocities * self.max_agents_speed
            self.simulation.agents.set_velocity(new_velocities)
            self.simulation.agents.move(self.time_step)

            # rewards
            dist = np.linalg.norm(self.simulation.agents.positions - self.simulation.agents.targets, axis=1)
            self.rewards = self.get_reward(distance=dist)

            # check if agent is still on scene
            out_of_bounds = (0 > self.simulation.agents.positions[:, 0]) | \
                            (self.simulation.agents.positions[:, 0] > self.win_size[0]) | \
                            (0 > self.simulation.agents.positions[:, 1]) | \
                            (self.simulation.agents.positions[:, 1] > self.win_size[1])
            self.rewards[out_of_bounds] -= self.out_of_bounds_penalty

            self.dones = (dist < np.finfo(float).eps) | out_of_bounds

        self.curr_agent_action = (self.curr_agent_action + 1) % self.agents_num

        # observation, reward, done, info
        return obs, curr_reward, is_done, {}

    def reset(self):
        self.time = 0

        self.curr_agent_action = 0
        self.next_step_velocities = np.empty(shape=(self.agents_num, 2))
        self.dones = np.full(shape=self.agents_num, fill_value=False, dtype=bool)
        self.rewards = np.zeros(shape=self.agents_num)

        self.initialize_simulation()
        distance = np.linalg.norm(self.simulation.agents.positions - self.simulation.agents.targets, axis=1)
        # get multiplicity of distance quantification
        self.closest_reached_distance = distance // self.distance_quantification * self.distance_quantification
        return self.get_observations(0)

    def render(self, mode='human', close=False):
        if mode == 'human':
            if not self.window:
                self.window = pg.display.set_mode(self.win_size)
                pg.display.set_caption("Collision Avoidance")
        self.window.fill((60, 60, 60))
        self.simulation.update(self.window)
        pg.display.update()

    def close(self):
        # pg.display.quit()
        super().close()

    def get_observations(self, agent_idx: int):
        nearest = self.simulation.agents.get_nearest_neighbours(self.visible_agents_num)[agent_idx]

        visible_positions = self.simulation.agents.positions[nearest]
        visible_velocities = self.simulation.agents.velocities[nearest]

        return {
            'velocity_x': visible_velocities[:, 0],
            'velocity_y': visible_velocities[:, 1],
            'position_x': visible_positions[:, 0],
            'position_y': visible_positions[:, 1],
            'target': self.simulation.agents.targets[agent_idx],
            'self_position': self.simulation.agents.positions[agent_idx]
        }

    def get_reward(self, distance: float):
        nearest = self.simulation.agents.get_nearest_neighbours(1)
        current_reward = np.full(self.agents_num, fill_value=-1.)  # time penalty

        # reward for reaching goal
        current_reward[distance < np.finfo(np.float32).eps] += self.goal_reward

        # reward for shortening distance to target
        reached = distance < self.closest_reached_distance
        self.closest_reached_distance[reached] -= self.distance_quantification
        current_reward[reached] += self.distance_reward * self.distance_quantification

        # penalty for collisions
        colliding = np.array([
            self.simulation.is_colliding(i, nearest[i, 0])
            for i, near in enumerate(nearest)
        ])
        current_reward[colliding] -= self.collision_penalty

        return current_reward


if __name__ == '__main__':
    env = MultiAgentCollisionAvoidanceEnv(
        agents_num=4,
        max_speed=32.,
        win_size=(600, 600)
    )

    print(f'actions: {env.action_space}')
    print(f'observations: {env.observation_space}')

    episodes_num = 5
    for _ in range(episodes_num):  # episodes
        done = np.full(env.agents_num, fill_value=False, dtype=bool)
        current_total_reward = 0
        observation = env.reset()
        t = 0
        while not all(done):
            env.render()
            observation, reward, done, _ = env.step(np.zeros(shape=(env.agents_num, 2), dtype=float))
            current_total_reward += reward
            t += 1

        print(f'ends after {t} with total reward of {current_total_reward}')
    env.close()
