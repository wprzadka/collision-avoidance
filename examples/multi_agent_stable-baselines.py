from stable_baselines3 import SAC
from openai_gym_env.env import CollisionAvoidanceEnv
from stable_baselines3.common.env_checker import check_env
import time


if __name__ == '__main__':

    env = CollisionAvoidanceEnv(10, 3, 10., win_size=(600, 400), algorithm='ORCA')
    check_env(env)

    model = SAC.load('sac_model.zip', env)
    # model.learn(total_timesteps=10)
    # model.save('sac_model')

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
            action, state = model.predict(observation, deterministic=True)
            observation, reward, done, _ = env.step(action)
            current_total_reward += reward
            t += 1
            time.sleep(0.05)

        print(f'ends after {t} with total reward of {current_total_reward}')
    env.close()
