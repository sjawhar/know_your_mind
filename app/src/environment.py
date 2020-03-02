import math
import numpy as np
import pickle


class Env(object):
    viewer = None

    def __init__(
        self,
        train_data,
        test_data,
        reward_model,
        len_max=128,
        num_features=64,
        num_classes=8,
        feature_indices=None,
    ):
        # we only have one attention bar, therefore, only have 1*2 table. [I,L]
        # I denotes Initial point, L denotes length
        self.bar = np.zeros(
            1,
            dtype=[("start", np.float32), ("end", np.float32), ("length", np.float32),],
        )
        self.bar["end"] = self.bar["start"] + self.bar["length"]
        self.bar["length"] = 64  # the length of attention bar
        self.bar["start"] = 0  # the initial point of attention bar
        self.beta = 0.1
        self.len_max = len_max
        self.min_length = 15
        self.num_actions = 4
        self.num_classes = num_classes
        self.num_features = num_features
        self.reward = 0
        self.reward_model = reward_model
        self.test_data = test_data
        self.train_data = train_data

        if feature_indices is None:
            feature_indices = np.random.choice(
                self.num_features, size=len_max, replace=True
            )
        self.feature_indices = feature_indices
        print("feature indices", feature_indices)

    def clip(self, dd):
        return np.clip(dd, a_min=0, a_max=self.len_max - 1)

    def featurize_data(self, data):
        start = int(self.bar["start"])
        end = int(self.bar["end"])
        feature_indices = self.feature_indices[start:end]
        return np.hstack([data[:, feature_indices], data[:, -1:]])

    def step(self, action):
        reward_raw = self.reward_model(
            self.featurize_data(self.train_data),
            test_data=self.featurize_data(self.test_data),
        )
        done = reward_raw > 0.5

        # Silhouette Coefficient reward ranges from [-1,1], reward+1 range [0,2]
        start = int(self.bar["start"])
        length = int(self.bar["length"])
        end = int(self.bar["end"])
        self.reward = (
            math.exp(reward_raw) / (math.exp(1) - 1)
            - self.beta * (end - start) / self.len_max
        )

        move = np.random.randint(1, high=15, size=1)
        # To caculate the next state from the current state
        if action == 0:
            # move left
            self.bar["start"] = self.clip(start - move)
            self.bar["end"] = self.clip(self.bar["start"] + length)
        elif action == 1:
            # move right
            self.bar["start"] = self.clip(start + move)
            self.bar["end"] = self.clip(self.bar["start"] + length)
        elif action == 2:
            # contract
            self.bar["start"] = self.clip(start + move)
            self.bar["end"] = self.clip(end - move)
            self.bar["length"] = self.bar["end"] - self.bar["start"]
        elif action == 3:
            # expand
            self.bar["start"] = self.clip(start - move)
            self.bar["end"] = self.clip(end + move)
            self.bar["length"] = self.bar["end"] - self.bar["start"]

        if self.bar["end"] - self.bar["start"] < self.min_length:
            self.bar["start"] = self.clip(self.bar["end"] - self.min_length)
            self.bar["end"] = self.bar["start"] + self.min_length
            self.bar["length"] = self.bar["end"] - self.bar["start"]

        return self.bar, self.reward, done, reward_raw, self.feature_indices

    def reset(self):
        self.bar["start"] = (self.len_max - self.num_features) / 2
        self.bar["length"] = self.num_features
        # self.bar['length'] = self.num_features+6  # for short features, e.g., only 3 dimensions

        self.bar["end"] = self.clip(self.bar["start"] + self.bar["length"])
        self.bar["start"] = self.clip(self.bar["start"])
        return self.bar

    def sample_action(self):
        return np.random.rand(2) - 0.5  # two radians
