from abc import ABC

import gym
from gym import spaces
import numpy as np
import pygame as pg

from environment.simulation import Simulation
from environment.agents import Agents


class CollisionAvoidanceEnv(gym.Env, ABC):
    metadata = {'render.modes': ['human']}

    def __init__(self, agents_num: int, visible_agents_num: int, max_speed: float):
        super(CollisionAvoidanceEnv, self).__init__()

        self.max_agents_speed = max_speed
        self.agents_num = agents_num
        self.visible_agents_num = visible_agents_num

        self.action_space = spaces.Box(
            shape=2,
            low=-max_speed,
            high=max_speed
        )

        self.observation_space = spaces.Box(
            shape=(visible_agents_num, 2, 2),
            low=-max_speed,
            high=max_speed
        )

        # rendering
        self.window = pg.display.set_mode()
        pg.display.set_caption("Collision Avoidance")
        # simulation
        self.simulation = Simulation()
        self.initialize_simulation(agents_num, pg.display.get_surface().get_size())
        self.time = 0

    def initialize_simulation(self, agents_num: int, win_size: tuple):
        agents = Agents(
            agents_num=agents_num,
            positions=np.random.rand(agents_num, 2) * win_size,
            # positions=np.array([
            #     [100., 200.],
            #     [400., 200.],
            #     [200., 300.],
            #     [400., 500.]
            # ]),
            radiuses=np.full((agents_num, 1), 10),
            max_speeds=np.full((agents_num, 1), 100.),
            desired_speeds=np.full((agents_num, 1), 75.),
            velocity_diff_range=np.full((agents_num, 1), 10.)
        )
        targets = np.random.rand(agents_num, 2) * win_size
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
        self.simulation.agents.velocities[0] = np.zeros(2)
        self.simulation.agents.move(0.01)

        visible_positions = self.simulation.agents.positions[1: self.visible_agents_num + 1]
        visible_velocities = self.simulation.agents.velocities[1: self.visible_agents_num + 1]
        observation = np.stack([visible_positions, visible_velocities], axis=1)

        distance = np.linalg.norm(self.simulation.agents.positions[0] - self.simulation.agents.targets[0])
        reward = -distance

        # observation, reward, done, debug
        return observation, reward, self.time > 1000, None

    def reset(self):
        self.time = 0
        self.initialize_simulation(self.agents_num, pg.display.get_surface().get_size())

    def render(self, mode='human', close=False):
        if mode == 'human':
            if not self.window:
                win_size = (1200, 900)
                self.window = pg.display.set_mode(win_size)
                pg.display.set_caption("Collision Avoidance")
        self.window.fill((60, 60, 60))
        self.simulation.update(self.window)
        pg.display.update()

    def close(self):
        # pg.display.quit()
        super().close()


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
            observation, reward, done, _ = env.step(None)
            current_total_reward += reward
            t += 1

        print(f'ends after {t} with total reward of {current_total_reward}')
    env.close()
