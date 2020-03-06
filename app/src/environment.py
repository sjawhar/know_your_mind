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
        self.bar = np.zeros(1, dtype=[("start", np.int32), ("end", np.int32)])
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
        self.set_state(0, int(len_max) // 2)

    def clip(self, index):
        return int(np.clip(index, a_min=0, a_max=self.len_max - 1))

    def featurize_data(self, data):
        start, end = self.get_state()
        feature_indices = list(self.feature_indices[start:end]) + [-1]
        return data[:, feature_indices]

    def get_state(self):
        return [int(x) for xs in self.bar for x in xs]

    def set_state(self, bar, end=None):
        if type(bar) is not list and end is not None:
            bar = [bar, end]
        self.bar["start"], self.bar["end"] = bar
        logger.debug(f"Set state to {self.bar}")

    def step(self, action):
        start, end = self.get_state()
        move = np.random.randint(1, high=15, size=1)
        if action == ACTION_CONTRACT:
            start = self.clip(start + move)
            end = self.clip(end - move)
        elif action == ACTION_EXPAND:
            start = self.clip(start - move)
            end = self.clip(end + move)
        elif action == ACTION_MOVE_LEFT:
            start = self.clip(start - move)
            end = self.clip(end - move)
        elif action == ACTION_MOVE_RIGHT:
            start = self.clip(start + move)
            end = self.clip(end + move)
        if end - start < self.min_length:
            start = self.clip(end - self.min_length)
            end = self.clip(start + self.min_length)

        self.set_state(start, end)

        reward_raw = self.reward_model(
            self.featurize_data(self.train_data),
            test_data=self.featurize_data(self.test_data),
        )
        done = reward_raw > self.reward_threshold
        self.reward = (
            math.exp(reward_raw) / (math.exp(1) - 1)
            - self.beta * (end - start) / self.len_max
        )

        return self.get_state(), self.reward, reward_raw, done

    def reset(self):
        start = self.clip((self.len_max - self.num_features) // 2)
        end = self.clip(start + self.num_features)
        self.set_state(start, end)
        return self.get_state()
