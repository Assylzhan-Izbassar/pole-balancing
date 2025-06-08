import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from q_learning_bins import plot_running_avg

class SGDRegressor:
    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = 10e-2

    def partial_fit(self, X, Y):
        self.w += self.lr * (Y - X.dot(self.w)).dot(X)

    def predict(self, X):
        return X.dot(self.w)

class FeatureTransformer:
    def __init__(self, env):
        observation_examples = np.random.random((20000, 4)) * 2 - 2
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        featurizer = FeatureUnion([
            ('rbf1', RBFSampler(gamma=.05, n_components=1000)),
            ('rbf2', RBFSampler(gamma=1., n_components=1000)),
            ('rbf3', RBFSampler(gamma=.5, n_components=1000)),
            ('rbf4', RBFSampler(gamma=.1, n_components=1000)),
        ])
        feature_examples = featurizer.fit_transform(
            scaler.transform(observation_examples)
        )

        self.dimensions = feature_examples.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)


class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer

        for i in range(env.action_space.n):
            model = SGDRegressor(feature_transformer.dimensions)
            self.models.append(model)

    def predict(self, s):
        # print('predict', s)
        X = self.feature_transformer.transform(np.atleast_2d(s))
        return np.stack([m.predict(X) for m in self.models]).T

    def update(self, s, a, G):
        # print('update', s)
        X = self.feature_transformer.transform(np.atleast_2d(s))
        self.models[a].partial_fit(X, [G])

    def sample_action(self, s, eps):
        if np.random.rand() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))

def play_one(env, model, eps, gamma):
    observation = env.reset()[0]
    done = False
    total_reward = 0
    iters = 0

    while not done and iters < 2000:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, truncate, info = env.step(action)
        # print(reward)
        if done or truncate:
            reward = -200

        next = model.predict(observation)
        # print('next', next)
        assert(next.shape == (1, env.action_space.n))

        G = reward + gamma * np.max(next)
        model.update(prev_observation, action, G)

        if reward == 1:
            total_reward += reward
        iters += 1
    return total_reward

def main():
    env = gym.make('CartPole-v1')
    ft = FeatureTransformer(env)
    model = Model(env, ft)
    gamma = 0.99

    N = 500
    total_rewards = np.empty(N)
    costs = np.empty(N)

    for n in range(N):
        eps = 1.0 / np.sqrt(n + 1)
        total_reward = play_one(env, model, eps, gamma)
        total_rewards[n] = total_reward

        if n % 100 == 0:
            print('Episode', n, 'total_reward', total_reward)

    print('Avg reward', np.mean(total_rewards[-100:]))
    print('Total steps', sum(total_rewards))

    plt.plot(total_rewards)
    plt.title('Total rewards')
    plt.show()

    plot_running_avg(total_rewards)


if __name__ == '__main__':
    main()
