import h5py
import numpy as np
import pickle
import scipy.io as sc
from agent import DeepQNetwork
from environment import Env
from reward import cnn_reward_model


def get_data(data_path, scale):
    with h5py.File(data_path, "r") as f:
        X = f["data"][:, :-1]
        Y = f["data"][:, -1]

    print(X.shape)
    num_features = X.shape[1]

    if scale is True:
        # z-score scaling
        X = preprocessing.scale(X[:, :-1])

    _, Y = np.unique(Y, return_inverse=True)
    data = np.hstack([X, Y.reshape(-1, 1)])
    return data, num_features, Y.max() + 1


def run_env(env, agent, best_reward=0, best_state=[], reward_history=[]):
    # start training
    MAX_EPISODES = 50
    MAX_EP_STEPS = 50
    for i in range(MAX_EPISODES):
        # initial state
        old_state = env.reset()
        for step in range(MAX_EP_STEPS):
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
            if (step > 200) and (step % 5 == 0):  # learn once for each 5 steps
                agent.learn()
            if done:
                print("This episode is done, start the next episode")
                break
            old_state = new_state
    return best_reward, best_state, reward_history


def main(data_path, scale=False):

    data, num_features, num_classes = get_data(data_path, scale)
    env = Env(
        data,
        lambda *args: cnn_reward_model(*args, size1=10, size2=100),
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

    # RL.plot_cost()
    pickle.dump(agent.cost_history, open("cost_history", "wb"))
    pickle.dump(reward_history, open("reward_history", "wb"))
    pickle.dump(best_state, open("best_state", "wb"))


if __name__ == "__main__":
    main("../data/eeg-s.h5")
