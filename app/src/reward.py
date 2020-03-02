import numpy as np
import pickle
import random
import tensorflow as tf
import time
from scipy import stats
from scipy.signal import butter, lfilter
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

TF_CONFIG = tf.ConfigProto()
TF_CONFIG.gpu_options.allow_growth = True

# this function is used to transfer one column label to one hot label
def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_) + 1)
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# the CNN code
def compute_accuracy(sess, network, X, Y_true, keep):
    xs, ys, keep_prob, _, _, output, _ = network

    Y_pred = sess.run(output, feed_dict={xs: X, keep_prob: keep})
    correct = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return sess.run(accuracy, feed_dict={xs: X, ys: Y_true, keep_prob: keep})


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_1x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="SAME")


def get_conv_network(
    num_features, num_classes, size1=10, size2=164, lambd=0.001, learning_rate=0.001
):
    xs = tf.placeholder(tf.float32, [None, num_features])
    ys = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 1, num_features, 1])

    input_shape = (num_features) * size1
    ## conv1 layer ##
    W_conv1 = weight_variable(
        [1, 2, 1, size1]
    )  # patch 5x5, in size is 1, out size is 8
    b_conv1 = bias_variable([size1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 1*64*2
    # h_pool1 = max_pool_1x2(h_conv1)                          # output size 1*32x2
    h_pool1_flat = tf.reshape(h_conv1, [-1, input_shape])

    ## conv2 layer ##
    # depth_2 = 80
    # W_conv2 = weight_variable([2,2, size1, depth_2]) # patch 5x5, in size 32, out size 64
    # b_conv2 = bias_variable([depth_2])
    # h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2) # output size 1*32*64
    # h_pool2 = max_pool_1x2(h_conv2)

    ## fc1 layer ##
    W_fc1 = weight_variable([input_shape, size2])
    b_fc1 = bias_variable([size2])
    h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## fc2 layer ##
    W_fc2 = weight_variable([size2, num_classes])
    b_fc2 = bias_variable([num_classes])
    prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # the error between prediction and real data
    l2 = lambd * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    cross_entropy = (
        tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys)
        )
        + l2
    )  # Softmax loss
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    return xs, ys, keep_prob, train_step, h_fc1, prediction, cross_entropy


def get_batches_and_test(data, num_features, num_batches=9, test_percentage=0.1):
    data_size = data.shape[0]
    shuffle_order = np.random.permutation(data_size)
    test_split = int(data_size * (1 - test_percentage))

    test_indices = shuffle_order[test_split:]
    X_test = data[test_indices, :-1]
    Y_test = one_hot(data[test_indices, -1:])

    train_indices = shuffle_order[: test_split - (test_split % num_batches)]
    X_train = data[train_indices, :-1].reshape(num_batches, -1, num_features)
    Y_train = one_hot(data[train_indices, -1:]).reshape(
        num_batches, X_train.shape[1], -1
    )
    batches = [(X_train[i], Y_train[i]) for i in range(num_batches)]

    return batches, X_test, Y_test


def cnn_reward_model(
    data,
    num_features,
    num_classes,
    keep=1,
    lambd=0.001,
    learning_rate=0.001,
    n_neighbors=1,
    num_batches=9,
    num_epochs=20,
    size1=10,
    size2=164,
    test_percentage=0.1,
):
    batches, X_test, Y_test = get_batches_and_test(
        data, num_features, num_batches=num_batches, test_percentage=test_percentage
    )
    network = get_conv_network(
        num_features,
        num_classes,
        size1=size1,
        size2=size2,
        lambd=lambd,
        learning_rate=learning_rate,
    )

    xs, ys, keep_prob, train_step, embedding, prediction, cross_entropy = network

    with tf.Session(config=TF_CONFIG) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, num_epochs + 1):
            for X_train, Y_train in batches:
                sess.run(
                    train_step, feed_dict={xs: X_train, ys: Y_train, keep_prob: keep},
                )
            if (epoch % 10) == 0:
                cost = sess.run(
                    cross_entropy, feed_dict={xs: X_test, ys: Y_test, keep_prob: keep}
                )
                t1 = time.clock()
                conv_acc = compute_accuracy(sess, network, X_train, Y_train, keep)
                t2 = time.clock()
                print("testing time", t2 - t1)
                print(
                    "epoch",
                    epoch,
                    ", train acc",
                    conv_acc,
                    ", test acc",
                    compute_accuracy(sess, network, X_test, Y_test, keep),
                    ", the cost is",
                    cost,
                )

        D_train = np.vstack(
            [
                sess.run(
                    embedding, feed_dict={xs: X_train, ys: Y_train, keep_prob: keep},
                )
                for X_train, Y_train in batches
            ]
        )
        D_test = sess.run(
            embedding, feed_dict={xs: X_test, ys: Y_test, keep_prob: keep}
        )

    Y_train = np.argmax(np.vstack([y for _, y in batches]), axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(D_train, Y_train)
    Y_pred = clf.predict(D_test)
    knn_result = clf.score(D_test, Y_test)
    print(knn_result)
    print(classification_report(Y_pred, Y_test, digits=4))

    return knn_result
