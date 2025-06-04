import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import numpy as np
import matplotlib.pyplot as plt


def get_action(s, w):
  return 1 if np.dot(s, w) > 0 else 0


def play_one_episode(env, params):
  observation = env.reset()[0] # here we take the first element from array
  done = False
  t = 0

  while not done and t < 10000:
    t += 1
    action = get_action(observation, params)
    observation, reward, done, truncated, info = env.step(action)
    if done or truncated:
      break

  return t


def play_multiple_episodes(env, T, params):
  episode_lengths = np.empty(T)

  for t in range(T):
    episode_lengths[t] = play_one_episode(env, params)

  avg_length = episode_lengths.mean()
  print(f'Average episode length: {avg_length}')
  return avg_length


def random_search(env):
  episode_lengths = []
  best = 0
  params = None

  for t in range(100):
    new_params = np.random.random(4) * 2 - 1
    # print('new params:', new_params)
    avg_length = play_multiple_episodes(env, 100, new_params)
    episode_lengths.append(avg_length)

    if avg_length > best:
      params = new_params
      best = avg_length

  return episode_lengths, params

if __name__ == '__main__':
  env = gym.make('CartPole-v1', render_mode="rgb_array")
  # observation = env.reset()[0]
  # new_params = np.random.random(4) * 2 - 1
  #
  # action = 1 if np.dot(observation, new_params) > 0 else 0
  # stepped = env.step(action)
  #
  # print(observation[0], new_params, action)
  # print(stepped)

  episode_lengths, params = random_search(env)
  plt.plot(episode_lengths)
  plt.show()

  print('***Final run with final weigths***')
  num_eval_episodes = 4
  # env = RecordVideo(env, video_folder="cartpole-agent", name_prefix="eval",
  #                   episode_trigger=lambda x: False)
  env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

  play_multiple_episodes(env, 100, params)

  env.close()