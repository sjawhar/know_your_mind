import logging
import math
import numpy as np
import pickle

logger = logging.getLogger(__name__)

ACTION_CONTRACT = 0
ACTION_EXPAND = 1
ACTION_MOVE_LEFT = 2
ACTION_MOVE_RIGHT = 3


class Env(object):
    viewer = None

    def __init__(
        self,
        train_data,
        test_data,
        reward_model,
        feature_indices=None,
        len_max=128,
        num_classes=8,
        num_features=64,
        reward_threshold=0.5,
    ):
        self.bar = np.zeros(1, dtype=[("start", np.float32), ("end", np.float32)],)
        self.bar["start"] = 0
        self.bar["end"] = self.bar["start"] + len_max // 2
        self.beta = 0.1
        self.len_max = len_max
        self.min_length = 10
        self.num_actions = 4
        self.num_classes = num_classes
        self.num_features = num_features
        self.reward = 0
        self.reward_model = reward_model
        self.reward_threshold = reward_threshold
        self.test_data = test_data
        self.train_data = train_data

        if feature_indices is None:
            feature_indices = np.random.choice(
                self.num_features, size=len_max, replace=True
            )
        self.feature_indices = feature_indices
        logger.debug(f"feature_indices - {feature_indices}")

    def clip(self, dd):
        return np.clip(dd, a_min=0, a_max=self.len_max - 1)

    def featurize_data(self, data):
        start = int(self.bar["start"])
        end = int(self.bar["end"])
        feature_indices = list(self.feature_indices[start:end]) + [-1]
        return data[:, feature_indices]

    def step(self, action):
        reward_raw = self.reward_model(
            self.featurize_data(self.train_data),
            test_data=self.featurize_data(self.test_data),
        )
        done = reward_raw > self.reward_threshold

        start = int(self.bar["start"])
        end = int(self.bar["end"])
        self.reward = (
            math.exp(reward_raw) / (math.exp(1) - 1)
            - self.beta * (end - start) / self.len_max
        )

        move = np.random.randint(1, high=15, size=1)
        if action == ACTION_CONTRACT:
            self.bar["start"] = self.clip(start + move)
            self.bar["end"] = self.clip(end - move)
        elif action == ACTION_EXPAND:
            self.bar["start"] = self.clip(start - move)
            self.bar["end"] = self.clip(end + move)
        elif action == ACTION_MOVE_LEFT:
            self.bar["start"] = self.clip(start - move)
            self.bar["end"] = self.clip(end - move)
        elif action == ACTION_MOVE_RIGHT:
            self.bar["start"] = self.clip(start + move)
            self.bar["end"] = self.clip(end + move)

        if self.bar["end"] - self.bar["start"] < self.min_length:
            self.bar["start"] = self.clip(self.bar["end"] - self.min_length)
            self.bar["end"] = self.clip(self.bar["start"] + self.min_length)

        return self.bar, self.reward, done, reward_raw, self.feature_indices

    def reset(self):
        self.bar["start"] = self.clip((self.len_max - self.num_features) // 2)
        self.bar["end"] = self.clip(self.bar["start"] + self.num_features)
        return self.bar
