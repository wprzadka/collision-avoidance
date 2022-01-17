from stable_baselines3 import SAC
from openai_gym_env.env import CollisionAvoidanceEnv
from stable_baselines3.common.env_checker import check_env


if __name__ == '__main__':

    env = CollisionAvoidanceEnv(5, 3, 10., win_size=(1200, 900), algorithm='VO')
    check_env(env)

    # model = SAC('MultiInputPolicy', env).learn(total_timesteps=10000)
    # model.save('sac_model')
    model = SAC.load('sac_model.zip')

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

        print(f'ends after {t} with total reward of {current_total_reward}')
    env.close()
