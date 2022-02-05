import numpy as np
from stable_baselines3 import PPO, SAC
from openai_gym_env.multi_agent_env import MultiAgentCollisionAvoidanceEnv
from stable_baselines3.common.env_checker import check_env
import time


if __name__ == '__main__':

    env = MultiAgentCollisionAvoidanceEnv(
        agents_num=3,
        visible_agents_num=2,
        max_speed=10.,
        win_size=(600, 400)
    )

    # model = SAC(env=env, policy='MultiInputPolicy')
    model = SAC.load('multi_sac_model.zip', env)
    # model.learn(total_timesteps=1000)
    # model.save('multi_sac_model')

    print(f'actions: {env.action_space}')
    print(f'observations: {env.observation_space}')

    episodes_num = 20
    for _ in range(episodes_num):  # episodes
        done = False
        current_total_reward = np.zeros(shape=env.agents_num)
        observation = env.reset()
        t = 0
        while not done:
            env.render()

            for i in range(env.agents_num):
                action, _ = model.predict(observation=observation, deterministic=True)
                observations, reward, done, _ = env.step(action)
                current_total_reward[i] += reward
            time.sleep(0.05)
            t += 1

        print(f'ends after {t} with total reward of {current_total_reward}')
    env.close()
