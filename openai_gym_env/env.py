from abc import ABC

import gym
from gym import spaces
import numpy as np
import pygame as pg

from environment.simulation import Simulation
from environment.agents import Agents


class CollisionAvoidanceEnv(gym.Env, ABC):
    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            agents_num: int,
            visible_agents_num: int,
            max_speed: float,
            win_size: tuple = (1200, 900)
    ):
        super(CollisionAvoidanceEnv, self).__init__()

        self.max_agents_speed = max_speed
        self.agents_num = agents_num
        self.visible_agents_num = visible_agents_num
        self.win_size = win_size

        self.action_space = spaces.Box(
            shape=(2,),
            low=-max_speed,
            high=max_speed
        )

        self.observation_space = spaces.Dict({
            'velocity': spaces.Box(
                shape=(visible_agents_num, 2),
                low=-max_speed,
                high=max_speed
            ),
            'position': spaces.Box(
                shape=(visible_agents_num, 2),
                low=np.zeros((visible_agents_num, 2)),
                high=np.full((visible_agents_num, 2), fill_value=self.win_size)
            ),
        })
        # simulation
        self.simulation = Simulation()
        self.initialize_simulation()
        self.time = 0
        # rendering
        self.window = None

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
        self.time += 1

        new_velocities = self.simulation.agents.get_preferred_velocities()
        self.simulation.agents.set_velocity(new_velocities)
        # override 1st agent velocity with action
        self.simulation.agents.velocities[0] = action
        self.simulation.agents.move(0.01)

        # observation
        obs = self.get_observations()
        # reward
        distance = np.linalg.norm(self.simulation.agents.positions[0] - self.simulation.agents.targets[0])
        reward = -distance

        # observation, reward, done, info
        return obs, reward, self.time > 1000, {}

    def reset(self):
        self.time = 0
        self.initialize_simulation()
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
        visible_positions = self.simulation.agents.positions[1: self.visible_agents_num + 1]
        visible_velocities = self.simulation.agents.velocities[1: self.visible_agents_num + 1]
        # assert all([all(a < b) for a, b in zip(visible_velocities, self.observation_space['velocity'].high)])
        return {
            'velocity': visible_velocities,
            'position': visible_positions,
        }


if __name__ == '__main__':
    env = CollisionAvoidanceEnv(5, 3, 10.)

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
