import h5py
import numpy as np
import pickle
import scipy.io as sc
from agent import DeepQNetwork
from environment import Env
from pathlib import Path
from reward import cnn_reward_model


def get_data(data_path, scale):
    data = {}
    Y = []
    dsets = ["train", "test"]
    with h5py.File(data_path, "r") as f:
        for dset in dsets:
            X = f[dset][:, :-1]
            Y += list(f[dset][:, -1])

            print(dset, X.shape)

            if scale is True:
                # z-score scaling
                X = preprocessing.scale(X)
            data[dset] = X

    num_features = data[dsets[0]].shape[1]

    _, Y = np.unique(Y, return_inverse=True)
    previous = 0
    for dset in dsets:
        X = data[dset]
        dset_len = len(X)
        data[dset] = np.hstack(
            [X, Y[previous : previous + dset_len].reshape(dset_len, 1)]
        )
        previous += dset_len

    test_data = data["test"]
    np.random.shuffle(test_data)
    return data["train"], test_data, num_features, Y.max() + 1


def run_env(
    env,
    agent,
    best_reward=0,
    best_state=[],
    episode_steps=50,
    episodes=50,
    reward_history=[],
):
    for i in range(episodes):
        # initial state
        old_state = env.reset()
        for step in range(1, episode_steps + 1):
            action = agent.choose_action(old_state)  #
            # print('episode', i, 'step', step, 'Action', action, 'state', state)

            new_state, reward, done, acc, indices = env.step(action, step, i)
            # print('new_state, reward, done, acc', new_state, reward, done, acc)
            # store the state, action, reward and the next state
            # print(old_state.shape, action.shape, reward.shape, new_state.shape)
            agent.store_transition(old_state, action, reward, indices, new_state)
            reward_history.append([i, step, reward])
            # record best reward
            current_reward = reward

            if current_reward > best_reward:
                best_reward = reward
                best_state = [acc, reward, old_state, new_state, indices]
            if (step >= 200) and (step % 5 == 0):  # learn once for each 5 steps
                agent.learn()
            if done:
                print("This episode is done, start the next episode")
                break
            old_state = new_state
    return best_reward, best_state, reward_history


def main(data_path, results_dir, scale=False):
    data, test_data, num_features, num_classes = get_data(data_path, scale)
    reward_model = lambda X, **kwargs: cnn_reward_model(
        X, num_classes, size1=10, size2=100, **kwargs
    )
    env = Env(
        data,
        reward_model,
        len_max=128,
        num_features=num_features,
        num_classes=num_classes,
    )
    agent = DeepQNetwork(
        e_greedy_increment=0.002,
        e_greedy=0.8,
        learning_rate=0.01,
        len_max=env.len_max,
        memory_size=2000,
        num_actions=env.num_actions,
        num_features=2,
        replace_target_iter=200,
        reward_decay=0.8,
        # output_graph=True,
    )
    best_reward, best_state, reward_history = run_env(env, agent)
    print(best_state, best_reward)

    _, _, _, attention, indices = best_state
    features = list(indices[attention["start"] : attention["end"]]) + [-1]
    test_acc = reward_model(
        data[:, features], test_data=test_data[:, features], test_percentage=0
    )
    print("Final accuracy:", test_acc)

    results_dir = Path(results_dir).resolve()
    if not results_dir.exists():
        results_dir.mkdir()
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="know_your_mind")
    parser.add_argument("DATA_FILE")
    parser.add_argument("RESULTS_DIR")

    args = parser.parse_args()

    main(args.DATA_FILE, args.RESULTS_DIR)
