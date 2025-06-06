import gymnasium as gym
import os
import sys

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gym import wrappers
from datetime import datetime

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor


class FeatureTransformer:
    def __init__(self, env, n_components=500):
        observation_examples = np.array([
            env.observation_space.sample() for _ in range(10000)
        ])
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        featurizer = FeatureUnion([
            ('rbf1', RBFSampler(gamma=5.0, n_components=n_components)),
            ('rbf2', RBFSampler(gamma=2.0, n_components=n_components)),
            ('rbf3', RBFSampler(gamma=1.0, n_components=n_components)),
            ('rbf4', RBFSampler(gamma=0.5, n_components=n_components)),
        ])
        # example_features = featurizer.fit_transform(scaler.transform(observation_examples))
        featurizer.fit(scaler.transform(observation_examples))

        # self.dimensions = example_features.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer


    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)


class Model:
    def __init__(self, env, feature_transformer, learning_rate):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer

        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate=learning_rate)
            model.partial_fit(
                feature_transformer.transform([env.reset()[0]]),
                [0]
            )
            self.models.append(model)

    def predict(self, s):
        X = self.feature_transformer.transform([s])
        # result = np.stack([m.predict(X) for m in self.models]).T
        result = np.array([m.predict(X) for m in self.models])
        # assert(len(result.shape) == 2)
        return result

    def update(self, s, a, G):
        X = self.feature_transformer.transform([s])
        assert(len(X.shape) == 2)
        self.models[a].partial_fit(X, [G])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))


def play_one(model, env, eps, gamma):
    observation = env.reset()[0]
    done = False
    total_reward = 0
    iters = 0

    while not done and iters < 10000:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, truncated, info = env.step(action)

        # if done or truncated:
        #     G = reward
        # else:
        #     Q_next = model.predict(prev_observation)
        #     G = reward + gamma * np.max(Q_next[0])

        G = reward + gamma * np.max(model.predict(observation)[0])
        model.update(prev_observation, action, G)

        total_reward += reward
        iters += 1

    return total_reward


def plot_cost_to_go(env, estimator, num_tiles=20):
    x = np.linspace(
        env.observation_space.low[0],
        env.observation_space.high[0],
        num=num_tiles
    )
    y = np.linspace(
        env.observation_space.low[1],
        env.observation_space.high[1],
        num=num_tiles
    )

    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(
        lambda _: -np.max(estimator.predict(_)),
        2,
        np.dstack([X, Y])
    )

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z,
                           rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm,
                           vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost-To-Go == -V(s)')
    ax.set_title('Cost-To-Go Function')

    fig.colorbar(surf)
    plt.show()


def plot_running_avg(total_rewards):
    N = len(total_rewards)
    running_avg = np.empty(N)

    for t in range(N):
        running_avg[t] = total_rewards[max(0, t-100):(t+1)].mean()

    plt.plot(running_avg)
    plt.title('Running Average')
    plt.show()


def main(show_plots=True):
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft, 'constant')
    gamma = 0.99

    N = 300
    total_rewards = np.empty(N)

    for n in range(N):
        eps = 1.0 / (0.1 * n + 1)

        if n == 199:
            print('eps', eps)

        total_reward = play_one(model, env, eps, gamma)
        total_rewards[n] = total_reward

        if (n + 1) % 10 == 0:
            print('Episode:', n, 'Total Reward:', total_reward)

    print('Avg reward for last 100 episodes:', total_rewards[-100:].mean())
    print('Total steps:', -total_rewards.sum())

    if show_plots:
        plt.plot(total_rewards)
        plt.title('Total Reward')
        plt.show()

        plot_running_avg(total_rewards)
        plot_cost_to_go(env, model)


if __name__ == '__main__':
    main()

