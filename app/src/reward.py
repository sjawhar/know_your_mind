import logging
import numpy as np
import tensorflow as tf
import time
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

TF_CONFIG = tf.ConfigProto()
TF_CONFIG.gpu_options.allow_growth = True

logger = logging.getLogger(__name__)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def get_conv_network(
    num_features,
    num_classes,
    conv_num_filters=10,
    fc_num_units=100,
    lambd=0.001,
    learning_rate=0.001,
):
    logger.debug(f"Building CNN...")
    xs = tf.placeholder(tf.float32, [None, num_features])
    ys = tf.placeholder(tf.int32, [None])
    drop = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 1, num_features, 1])

    ## conv1 layer ##
    W_conv1 = weight_variable([1, 2, 1, conv_num_filters])
    b_conv1 = bias_variable([conv_num_filters])
    h_conv1 = tf.nn.relu(
        tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1
    )

    input_shape = num_features * conv_num_filters
    h_conv1_flat = tf.reshape(h_conv1, [-1, input_shape])

    ## fc1 layer ##
    W_fc1 = weight_variable([input_shape, fc_num_units])
    b_fc1 = bias_variable([fc_num_units])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv1_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, rate=drop)

    # TODO: Handle binary case
    ## fc2 layer ##
    W_fc2 = weight_variable([fc_num_units, num_classes])
    b_fc2 = bias_variable([num_classes])
    prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    ## loss ##
    l2 = lambd * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    loss = (
        tf.math.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=ys)
        )
        + l2
    )
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    logger.debug(f"CNN built!")
    return xs, ys, drop, train_step, h_fc1, prediction, loss


def get_batches_and_test(data, num_features, batch_size=128, test_percentage=0.1):
    logger.debug("Batching and splitting data...")
    data_size = data.shape[0]
    shuffle_order = np.random.permutation(data_size)
    test_split = int(data_size * (1 - test_percentage))
    num_batches = test_split // batch_size

    test_indices = shuffle_order[test_split:]
    train_indices = shuffle_order[: test_split - (test_split % batch_size)]

    batches = data[train_indices].reshape(num_batches, batch_size, -1)

    logger.debug(f"{data_size} samples")
    logger.debug(f"{len(train_indices)} train - {num_batches} batches of {batch_size}")
    logger.debug(f"{len(test_indices)} test ({test_percentage*100}%)")
    return batches, data[test_indices]


def cnn_reward_model(
    data,
    num_classes,
    batch_size=128,
    conv_num_filters=10,
    dropout=0.2,
    fc_num_units=100,
    lambd=0.001,
    learning_rate=0.001,
    n_neighbors=1,
    num_epochs=20,
    test_data=None,
    test_epoch_frequency=10,
    test_percentage=0,
):
    logger.debug("Building CNN reward model...")
    num_features = data.shape[1] - 1
    g = tf.Graph()
    with g.as_default():
        network = get_conv_network(
            num_features,
            num_classes,
            conv_num_filters=conv_num_filters,
            fc_num_units=fc_num_units,
            lambd=lambd,
            learning_rate=learning_rate,
        )

    xs, ys, drop, train_step, embedding, prediction, loss = network
    batches, test_split_data = get_batches_and_test(
        data, num_features, batch_size=batch_size, test_percentage=test_percentage
    )
    if test_data is None:
        logger.debug("Using split data as test set")
        test_data = test_split_data
    X_test, Y_test = test_data[:, :-1], test_data[:, -1]

    logger.debug("Starting training...")
    with tf.Session(config=TF_CONFIG, graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, num_epochs + 1):
            logger.debug(f"Starting epoch {epoch}...")
            for batch_num in range(len(batches)):
                X_train, Y_train = batches[batch_num, :, :-1], batches[batch_num, :, -1]
                sess.run(
                    train_step, feed_dict={xs: X_train, ys: Y_train, drop: dropout},
                )
            if (test_epoch_frequency > 0) and (epoch % test_epoch_frequency) == 0:
                logger.info(f"EPOCH {epoch}")
                for set_name, X, Y in [
                    ("Train", X_train, Y_train),
                    ("Test", X_test, Y_test),
                ]:
                    t1 = time.clock()
                    Y_pred, conv_loss = sess.run(
                        [prediction, loss], feed_dict={xs: X, ys: Y, drop: 0}
                    )
                    t2 = time.clock()
                    conv_acc = np.mean(np.argmax(Y_pred, axis=1) == Y)
                    logger.info(f"{set_name} CNN accuracy - {conv_acc}")
                    logger.info(f"{set_name} CNN loss - {conv_loss}")
                    logger.info(f"{set_name} CNN prediction time - {t2 - t1}")

        logger.debug("Training finished!")
        logger.debug("Getting embeddings for training set...")
        X_train = batches[:, :, :-1].reshape(-1, num_features)
        Y_train = np.ravel(batches[:, :, -1])
        D_train = sess.run(embedding, feed_dict={xs: X_train, drop: 0})

        logger.debug("Getting embeddings for test set...")
        D_test = sess.run(embedding, feed_dict={xs: X_test, drop: 0})

    logger.debug("Training KNN classifier...")
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(D_train, Y_train)
    logger.debug(f"Predicting on test set ({D_test.shape[0]} samples)...")
    t1 = time.clock()
    Y_pred = clf.predict(D_test)
    t2 = time.clock()
    logger.debug("Calculating accuracy...")
    acc = accuracy_score(Y_test, Y_pred)
    logger.info(f"Test KNN accuracy - {acc}")
    logger.info(f"Test KNN prediction time - {t2 - t1}")
    logger.info("\n" + classification_report(Y_test, Y_pred, digits=4))

    del g
    return acc
