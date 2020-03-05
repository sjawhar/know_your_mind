"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import itertools

logger = logging.getLogger(__name__)
# np.random.seed(1)
# tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
        self,
        batch_size=32,
        dueling=True,
        e_greedy_increment=None,
        e_greedy=0.2,
        learning_rate=0.01,
        len_max=1000,
        memory_size=500,
        num_actions=4,
        num_features=2,
        output_graph=False,
        replace_target_iter=300,
        reward_decay=0.8,
    ):

        self.batch_size = batch_size
        self.cost_history = []
        self.dueling = dueling
        self.epsilon = 0 if e_greedy_increment is not None else e_greedy
        self.epsilon_increment = e_greedy_increment
        self.epsilon_max = e_greedy
        self.gamma = reward_decay
        self.learn_step_counter = 0
        self.len_max = len_max
        self.lr = learning_rate
        self.memory_size = memory_size
        self.num_actions = num_actions
        self.num_features = num_features
        self.replace_target_iter = replace_target_iter

        # [state_old, a, r, indices, state_new]
        self.memory = np.zeros((self.memory_size, 2 * num_features + 2 + self.len_max))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection("target_net_params")
        e_params = tf.get_collection("eval_net_params")
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    # Two nets have the same structure but different parameters.
    # One with parameters eval_net_params, another with target_net_params.
    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope("l1"):
                w1 = tf.get_variable(
                    "w1",
                    [self.num_features, n_l1],
                    initializer=w_initializer,
                    collections=c_names,
                )
                b1 = tf.get_variable(
                    "b1", [1, n_l1], initializer=b_initializer, collections=c_names
                )
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            if self.dueling:
                # Dueling DQN
                with tf.variable_scope("Value"):
                    w2 = tf.get_variable(
                        "w2", [n_l1, 1], initializer=w_initializer, collections=c_names
                    )
                    b2 = tf.get_variable(
                        "b2", [1, 1], initializer=b_initializer, collections=c_names
                    )
                    self.V = tf.matmul(l1, w2) + b2

                with tf.variable_scope("Advantage"):
                    w2 = tf.get_variable(
                        "w2",
                        [n_l1, self.num_actions],
                        initializer=w_initializer,
                        collections=c_names,
                    )
                    b2 = tf.get_variable(
                        "b2",
                        [1, self.num_actions],
                        initializer=b_initializer,
                        collections=c_names,
                    )
                    self.A = tf.matmul(l1, w2) + b2

                with tf.variable_scope("Q"):
                    out = self.V + (
                        self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True)
                    )  # Q = V(s) + A(s,a)
            else:
                with tf.variable_scope("Q"):
                    w2 = tf.get_variable(
                        "w2",
                        [n_l1, self.num_actions],
                        initializer=w_initializer,
                        collections=c_names,
                    )
                    b2 = tf.get_variable(
                        "b2",
                        [1, self.num_actions],
                        initializer=b_initializer,
                        collections=c_names,
                    )
                    out = tf.matmul(l1, w2) + b2

            return out

        # ------------------ build evaluate_net ------------------
        self.state_old = tf.placeholder(
            tf.float32, [None, self.num_features], name="state_old"
        )  # input
        self.q_target = tf.placeholder(
            tf.float32, [None, self.num_actions], name="Q_target"
        )  # for calculating loss

        with tf.variable_scope("eval_net"):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = (
                ["eval_net_params", tf.GraphKeys.GLOBAL_VARIABLES],
                32,
                tf.random_normal_initializer(0.0, 0.3),
                tf.constant_initializer(0.1),
            )  # config of layers, 0 is mean, 0.3 is stddev

            # first layer. collections is used later when assign to target net
            with tf.variable_scope("l1"):
                w1 = tf.get_variable(
                    "w1",
                    [self.num_features, n_l1],
                    initializer=w_initializer,
                    collections=c_names,
                )
                b1 = tf.get_variable(
                    "b1", [1, n_l1], initializer=b_initializer, collections=c_names
                )
                l1 = tf.nn.relu(tf.matmul(self.state_old, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope("l2"):
                w2 = tf.get_variable(
                    "w2",
                    [n_l1, self.num_actions],
                    initializer=w_initializer,
                    collections=c_names,
                )
                b2 = tf.get_variable(
                    "b2",
                    [1, self.num_actions],
                    initializer=b_initializer,
                    collections=c_names,
                )
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope("loss"):
            # loss = [(q_target-q_eval)^2]/n where n = len(q_target)
            self.loss = tf.reduce_mean(
                tf.math.squared_difference(self.q_target, self.q_eval)
            )
        with tf.variable_scope("train"):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.state_new = tf.placeholder(
            tf.float32, [None, self.num_features], name="state_new"
        )

        with tf.variable_scope("target_net"):
            # c_names(collections_names) are the collections to store variables
            c_names = ["target_net_params", tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope("l1"):
                w1 = tf.get_variable(
                    "w1",
                    [self.num_features, n_l1],
                    initializer=w_initializer,
                    collections=c_names,
                )
                b1 = tf.get_variable(
                    "b1", [1, n_l1], initializer=b_initializer, collections=c_names
                )
                l1 = tf.nn.relu(tf.matmul(self.state_new, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope("l2"):
                w2 = tf.get_variable(
                    "w2",
                    [n_l1, self.num_actions],
                    initializer=w_initializer,
                    collections=c_names,
                )
                b2 = tf.get_variable(
                    "b2",
                    [1, self.num_actions],
                    initializer=b_initializer,
                    collections=c_names,
                )
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(
        self, state_old, a, r, indices, state_new,
    ):
        if not hasattr(self, "memory_counter"):
            self.memory_counter = 0

        state_old = np.asarray([x for xs in state_old for x in xs])
        state_new = np.array([x for xs in state_new for x in xs])
        logger.debug(f"state_old - {state_old}")
        logger.debug(f"[a, r] - {[a, r]}")
        logger.debug(f"state_new - {state_new}")
        transition = np.hstack((state_old, [a, r], indices, state_new))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, state):
        # to have batch dimension when feed into tf placeholder
        short_state = np.zeros(1, dtype=[("start", np.float32), ("end", np.float32)])
        short_state["start"] = state["start"]
        short_state["end"] = state["end"]

        short_state = np.asarray([x for xs in short_state for x in xs])
        short_state = short_state[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the state and get q value for every actions

            actions_value = self.sess.run(
                self.q_eval, feed_dict={self.state_old: short_state}
            )
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.num_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            logger.info("\ntarget_params_replaced\n")

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.state_new: batch_memory[:, -self.num_features :],  # fixed params
                self.state_old: batch_memory[:, : self.num_features],  # newest params
            },
        )

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.num_features].astype(int)
        reward = batch_memory[:, self.num_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(
            q_next, axis=1
        )  # q_next is q_target

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.state_old: batch_memory[:, : self.num_features],
                self.q_target: q_target,
            },
        )
        self.cost_history.append(self.cost)

        if self.epsilon_increment != None and self.epsilon < self.epsilon_max:
            self.epsilon = min(self.epsilon + self.epsilon_increment, self.epsilon_max)
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt

        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel("Cost")
        plt.xlabel("training steps")
        plt.show()
