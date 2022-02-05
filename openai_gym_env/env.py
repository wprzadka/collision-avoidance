from abc import ABC

import gym
from gym import spaces
import numpy as np
import pygame as pg

from environment.simulation import Simulation
from environment.agents import Agents
from VO.ReciprocalVelocityObstacle import ReciprocalVelocityObstacle
from ORCA.OptimalReciprocalColisionAvoidance import ORCA


class CollisionAvoidanceEnv(gym.Env, ABC):
    metadata = {'render.modes': ['human']}
    available_algorithms = [None, 'VO', 'RVO', 'ORCA']
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
            algorithm: str = None,
            distance_quantification: int = 5,
            time_step: float = 0.25,
            time_limit: int = 100
    ):
        super(CollisionAvoidanceEnv, self).__init__()

        self.max_agents_speed = max_speed
        self.agents_num = agents_num
        self.visible_agents_num = visible_agents_num
        self.win_size = win_size

        self.action_space = spaces.Box(
            shape=(2,),
            low=-1,
            high=1
        )

        self.observation_space = spaces.Dict({
            'velocity_x': spaces.Box(
                shape=(visible_agents_num,),
                low=-max_speed,
                high=max_speed
            ),
            'velocity_y': spaces.Box(
                shape=(visible_agents_num,),
                low=-max_speed,
                high=max_speed
            ),
            'position_x': spaces.Box(
                shape=(visible_agents_num,),
                low=0,
                high=self.win_size[0]
            ),
            'position_y': spaces.Box(
                shape=(visible_agents_num,),
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

        if algorithm is None:
            self.get_velocities = self.simulation.agents.get_preferred_velocities
        elif algorithm in ['VO', 'RVO']:
            self.algorithm = ReciprocalVelocityObstacle(
                agents_num=self.agents_num,
                visible_agents_num=visible_agents_num,
                reciprocal=algorithm == 'RVO'
            )
            self.get_velocities = lambda: self.algorithm.compute_velocities(self.simulation.agents)
        elif algorithm == 'ORCA':
            self.algorithm = ORCA(
                agents_num=self.agents_num,
                visible_agents_num=visible_agents_num,
                time_step=time_step,
                time_horizon=20 * time_step
            )
            self.get_velocities = lambda: self.algorithm.compute_velocities(self.simulation.agents)
        else:
            raise Exception(f'{algorithm} is not in available algorithms')

    def initialize_simulation(self):
        agents = Agents(
            agents_num=self.agents_num,
            positions=np.random.rand(self.agents_num, 2) * self.win_size,
            # positions=np.array([
            #     [100., 200.],
            #     [400., 200.],
            #     [200., 300.],
            #     [400., 500.]
            # ]),
            radiuses=np.full((self.agents_num, 1), 10),
            max_speeds=np.full((self.agents_num, 1), self.max_agents_speed),
            desired_speeds=np.full((self.agents_num, 1), 0.75 * self.max_agents_speed),
            velocity_diff_range=np.full((self.agents_num, 1), 10.)
        )
        targets = np.random.rand(self.agents_num, 2) * self.win_size
        # targets = np.array([
        #     [400., 500.],
        #     [200., 500.],
        #     [500., 400.],
        #     [100., 200.]
        # ])
        self.simulation.initialize(agents, targets)

    def step(self, action):
        self.time += self.time_step

        new_velocities = self.get_velocities()
        # override 1st agent velocity with action
        new_velocities[0] = action * self.max_agents_speed

        self.simulation.agents.set_velocity(new_velocities)
        self.simulation.agents.move(self.time_step)

        # observation
        obs = self.get_observations()
        # reward
        dist = np.linalg.norm(self.simulation.agents.positions[0] - self.simulation.agents.targets[0])
        reward = self.get_reward(distance=dist)

        # check if agent is still on scene
        out_of_bounds = any(0 > self.simulation.agents.positions[0]) or \
                        any(self.simulation.agents.positions[0] > self.win_size)
        if out_of_bounds:
            reward -= self.out_of_bounds_penalty

        is_done = dist < np.finfo(float).eps or self.time > self.time_limit or out_of_bounds

        # observation, reward, done, info
        return obs, reward, is_done, {}

    def reset(self):
        self.time = 0
        self.initialize_simulation()
        distance = np.linalg.norm(self.simulation.agents.positions[0] - self.simulation.agents.targets[0])
        # get multiplicity of distance quantification
        self.closest_reached_distance = distance // self.distance_quantification * self.distance_quantification
        return self.get_observations()

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

    def get_observations(self):
        nearest = self.simulation.agents.get_nearest_neighbours(self.visible_agents_num)[0]

        visible_positions = self.simulation.agents.positions[nearest]
        visible_velocities = self.simulation.agents.velocities[nearest]
        # assert all([all(a < b) for a, b in zip(visible_velocities, self.observation_space['velocity'].high)])
        return {
            'velocity_x': visible_velocities[:, 0],
            'velocity_y': visible_velocities[:, 1],
            'position_x': visible_positions[:, 0],
            'position_y': visible_positions[:, 1],
            'target': self.simulation.agents.targets[0],
            'self_position': self.simulation.agents.positions[0]
        }

    def get_reward(self, distance: float):
        nearest = self.simulation.agents.get_nearest_neighbours(1)
        current_reward = -1  # time penalty

        # reward for reaching goal
        if distance < np.finfo(np.float32).eps:
            current_reward += self.goal_reward

        # reward for shortening distance to target
        if distance < self.closest_reached_distance:
            self.closest_reached_distance -= self.distance_quantification
            current_reward += self.distance_reward * self.distance_quantification

        # penalty for collisions
        if self.simulation.is_colliding(0, nearest[0]):
            current_reward -= self.collision_penalty

        return current_reward


if __name__ == '__main__':
    env = CollisionAvoidanceEnv(
        agents_num=20,
        visible_agents_num=3,
        max_speed=32.,
        algorithm='VO',
        win_size=(600, 600)
    )

    print(f'actions: {env.action_space}')
    print(f'observations: {env.observation_space}')

    episodes_num = 5
    for _ in range(episodes_num):  # episodes
        done = False
        current_total_reward = 0
        observation = env.reset()
        t = 0
        while not done:
            env.render()
            observation, reward, done, _ = env.step([0, 0])
            current_total_reward += reward
            t += 1

        print(f'ends after {t} with total reward of {current_total_reward}')
    env.close()
