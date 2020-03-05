import h5py
import logging
import numpy as np
import pickle
import scipy.io as sc
from .agent import DeepQNetwork
from .environment import Env
from .reward import cnn_reward_model
from pathlib import Path
from sklearn import preprocessing


logger = logging.getLogger(__name__)


def get_data(data_path, scale):
    data = {}
    Y = []
    dsets = ["train", "test"]
    with h5py.File(data_path, "r") as f:
        for dset in dsets:
            X = f[dset][:, :-1]
            Y += list(f[dset][:, -1])

            logger.debug(f"{dset} shape - {X.shape}")

            if scale is True:
                X = preprocessing.scale(X)
            data[dset] = X
            num_features = X.shape[1]

    _, Y = np.unique(Y, return_inverse=True)
    num_classes = Y.max() + 1

    previous = 0
    for dset in dsets:
        X = data[dset]
        dset_len = len(X)
        data[dset] = np.hstack([X, Y[previous : previous + dset_len].reshape(-1, 1)])
        previous += dset_len

    return data["train"], data["test"], num_features, num_classes


def run_env(
    env,
    agent,
    best_reward=0,
    best_state=[],
    episode_steps=50,
    episodes=50,
    reward_history=[],
    warmup=200,
):
    step_absolute = 0
    for episode in range(1, episodes + 1):
        state_new = env.reset()
        for step in range(1, episode_steps + 1):
            logger.info("==========")
            logger.info(f"EPISODE {episode} STEP {step}")
            logger.info("==========")
            step_absolute += 1
            state_old = np.copy(state_new)
            action = agent.choose_action(state_old)

            state_new, reward, done, reward_raw, indices = env.step(action)
            logger.debug(f"state_old - {state_old}")
            logger.debug(f"action - {action}")
            logger.debug(f"state_new - {state_new}")
            logger.debug(f"reward - {reward}")
            logger.debug(f"reward_raw - {reward_raw}")
            logger.debug(f"done - {done}")

            agent.store_transition(state_old, action, reward, indices, state_new)
            reward_history.append([episode, step, reward])

            if reward > best_reward:
                best_reward = reward
                best_state = [reward_raw, reward, state_old, state_new, indices]
            if (step_absolute >= warmup) and (step_absolute % 5 == 0):
                agent.learn()
            if done:
                logger.info("This episode is done, start the next episode")
                break
    return best_reward, best_state, reward_history


def run(
    data_path,
    results_dir,
    batch_size=128,
    dropout=0.2,
    reward_threshold=0.5,
    scale=False,
    test_epoch_frequency=10,
    warmup=200,
):
    train_data, test_data, num_features, num_classes = get_data(data_path, scale)
    reward_model = lambda X, **kwargs: cnn_reward_model(
        X,
        num_classes,
        batch_size=batch_size,
        conv_num_filters=10,
        dropout=dropout,
        fc_num_units=100,
        test_epoch_frequency=test_epoch_frequency,
        **kwargs,
    )
    env = Env(
        train_data,
        test_data,
        reward_model,
        len_max=128,
        num_features=num_features,
        num_classes=num_classes,
        reward_threshold=reward_threshold,
    )
    agent = DeepQNetwork(
        e_greedy_increment=0.002,
        e_greedy=0.2,
        learning_rate=0.01,
        len_max=env.len_max,
        memory_size=2000,
        num_actions=env.num_actions,
        num_features=2,
        replace_target_iter=200,
        reward_decay=0.8,
    )
    best_reward, best_state, reward_history = run_env(env, agent, warmup=warmup)
    _, _, _, attention, indices = best_state
    features = list(indices[int(attention["start"]) : int(attention["end"])]) + [-1]
    test_acc = reward_model(
        train_data[:, features], test_data=test_data[:, features], test_percentage=0,
    )
    logger.info(f"Best state - {best_state}")
    logger.info(f"Best reward - {best_reward}")
    logger.info(f"Final accuracy - {test_acc}")

    results_dir = Path(results_dir).resolve()
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    with open(results_dir / "results.pickle", "wb") as f:
        pickle.dump(
            {
                "best_state": best_state,
                "cost_history": agent.cost_history,
                "reward_history": reward_history,
                "test_acc": test_acc,
            },
            f,
        )
