# Name: Supriyo Ghosh
# Student ID: 215318728
# Assignment: 2

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def dropout(x, is_training, p):
    return tf.cond(is_training, lambda: tf.nn.dropout(x, p, name='dropout'), lambda: tf.identity(x))

# Data loader for notMNIST dataset
def load_notmnist_data():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        t = np.zeros((trainTarget.shape[0], 10))
        t[np.arange(trainTarget.shape[0]), trainTarget] = 1
        trainTarget = t
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        t = np.zeros((validTarget.shape[0], 10))
        t[np.arange(validTarget.shape[0]), validTarget] = 1
        validTarget = t
        testData, testTarget = Data[16000:], Target[16000:]
        t = np.zeros((testTarget.shape[0], 10))
        t[np.arange(testTarget.shape[0]), testTarget] = 1
        testTarget = t
        return (trainData.reshape(trainData.shape[0], -1), trainTarget, validData.reshape(validData.shape[0], -1), validTarget, testData.reshape(testData.shape[0], -1), testTarget)


def FCN(x, depth, name, use_dropout=False, is_training=tf.constant(False), use_relu=False):
    W = tf.get_variable(name=name + "_W", shape=(x.shape[1], depth), dtype=tf.float64)
    b = tf.get_variable(name=name+ "_b", shape=(depth,), dtype=tf.float64, initializer=tf.zeros_initializer)
    if use_dropout:
        if use_relu:
            return dropout(tf.nn.relu(tf.matmul(x, W) + b), is_training, 0.5)
        else:
            return dropout(tf.matmul(x, W) + b, is_training, 0.5)
    else:
        if use_relu:
            return tf.nn.relu(tf.matmul(x, W) + b)
        else:
            tf.matmul(x, W) + b

def build_network(input_node, is_training_t):

    #can be changed for 2-layer networks
    num_hidden_units = 1000
    L1_out = FCN(input_node[0], num_hidden_units, name='Layer_1', use_dropout=True, is_training=is_training_t,
                 use_relu=True)
    W = tf.get_variable(name="output_W", shape=(L1_out.shape[1], 10), dtype=tf.float64)
    b = tf.get_variable(name="output_b", shape=(10,), dtype=tf.float64, initializer=tf.zeros_initializer)

    #for multiple nerual networks
    # L1_out = FCN(input_node[0], num_hidden_units, name='Layer_1', use_dropout=True, is_training=is_training_t, use_relu=True)
    # L2_out = FCN(L1_out, num_hidden_units, name='Layer_2', use_dropout=True, is_training=is_training_t,
    #             use_relu=True)
    # L3_out = FCN(L2_out, num_hidden_units, name='Layer_3', use_dropout=True, is_training=is_training_t,
    #             use_relu=True)
    # L4_out = FCN(L3_out, num_hidden_units, name='Layer_4', use_dropout=True, is_training=is_training_t,
    #             use_relu=True)
    # W = tf.get_variable(name="output_W", shape=(L1_out.shape[1], 10), dtype=tf.float64)
    # b = tf.get_variable(name="output_b", shape=(10,), dtype=tf.float64, initializer=tf.zeros_initializer)

    # y_pred_raw = tf.matmul(L4_out, W) + b

    y_pred_raw = tf.matmul(L1_out, W) + b
    return y_pred_raw

def learning():
    xTrain, yTrain, xValid, yValid, xTest, yTest = load_notmnist_data()

    with tf.Graph().as_default():
        num_hidden_units = 1000
        decay = 0
        B = 500
        learning_rates = [0.01, 0.005, 0.001]
        iters = 5000
        max_num_epochs = (B*iters)//len(xTrain)
        if B*iters % len(xTrain):
            max_num_epochs += 1
        num_iters_per_epoch = len(xTrain) // B  # number of iterations we have to do for one epoch
        print("Num epochs = ", iters / num_iters_per_epoch)

        # hyperparameters
        learning_rate = tf.placeholder(dtype=tf.float64, name="learning-rate")
        is_training_t = tf.placeholder(dtype=tf.bool, name="is_training")

        base_iterator = tf.data.Iterator.from_structure((tf.float64, tf.float64), ((None, 784), (None, 10)))
        input_node = base_iterator.get_next()
        y_pred_raw = build_network(input_node, is_training_t)

        y_pred = tf.nn.softmax(y_pred_raw)
        CE_loss = tf.losses.softmax_cross_entropy(input_node[1], y_pred_raw)

        vars = tf.global_variables()
        l2s = []
        for var in vars:
            l2s.append(tf.nn.l2_loss(var))
        l2_loss = tf.reduce_sum(tf.stack(l2s, axis=0))
        total_loss = CE_loss + decay * l2_loss
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(input_node[1], 1)), tf.float32))

        # optimizer function
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

        X = tf.placeholder(dtype=tf.float64, name="X")
        Y = tf.placeholder(dtype=tf.float64, name="Y")
        Xdata = tf.data.Dataset.from_tensor_slices(X)
        Ydata = tf.data.Dataset.from_tensor_slices(Y)
        sample_dataset = tf.data.Dataset.zip((Xdata, Ydata))
        batched_dataset = sample_dataset.batch(B)
        # TODO run a lot of iterations, plot loss vs epochs and classification error vs epochs
        accuracy_list = []
        ce_list = []
        check_points = [iters//4, iters//2, 3*iters//4, iters-1]
        saver = tf.train.Saver(vars)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # initialize data input pippeline for training
            dataset_init = base_iterator.make_initializer(batched_dataset)
            for i in range(max_num_epochs):
                sess.run(dataset_init, feed_dict={X:xTrain, Y:yTrain})
                j = 0
                while True:
                    try:
                        sess.run([optimizer, ], feed_dict={learning_rate: 0.005, is_training_t: True})
                        j += 1
                        if i * num_iters_per_epoch + j in check_points:
                            saver.save(sess, '.\my_model', global_step=i)
                    except tf.errors.OutOfRangeError:
                        break

                # initialize data iterator for getting numbers to plot
                # on train
                sess.run(dataset_init, feed_dict={X: xTrain, Y: yTrain})
                this_acc = 0.0
                this_ce = 0.0
                j = 0
                while True:
                    try:
                        acc, ce = sess.run([accuracy, CE_loss], feed_dict={is_training_t: False})
                        this_acc += acc
                        this_ce += ce
                        j += 1
                    except tf.errors.OutOfRangeError:
                        break
                train_acc = this_acc/j
                train_ce = this_ce/j
                # on val
                sess.run(dataset_init, feed_dict={X: xValid, Y: yValid})
                this_acc = 0.0
                this_ce = 0.0
                j = 0
                while True:
                    try:
                        acc, ce = sess.run([accuracy, CE_loss], feed_dict={is_training_t: False})
                        this_acc += acc
                        this_ce += ce
                        j += 1
                    except tf.errors.OutOfRangeError:
                        break
                val_acc = this_acc / j
                val_ce = this_ce / j
                # on test
                sess.run(dataset_init, feed_dict={X: xTest, Y: yTest})
                this_acc = 0.0
                this_ce = 0.0
                j = 0
                while True:
                    try:
                        acc, ce = sess.run([accuracy, CE_loss], feed_dict={is_training_t: False})
                        this_acc += acc
                        this_ce += ce
                        j += 1
                    except tf.errors.OutOfRangeError:
                        break
                test_acc = this_acc / j
                test_ce = this_ce / j
                accuracy_list.append((train_acc, val_acc, test_acc))
                ce_list.append((train_ce, val_ce, test_ce))
                print("Epoch: {}, Training Loss: {}, Accuracies: [{}, {}, {}]".format(i,
                                                                                    train_ce, train_acc, val_acc,
                                                                                    test_acc))
    return accuracy_list, ce_list


def visualization(filepath, index=1):
    base_iterator = tf.data.Iterator.from_structure((tf.float64, tf.float64), ((None, 784), (None, 10)))
    input_node = base_iterator.get_next()
    is_training_t = tf.placeholder(dtype=tf.bool, name="is_training")
    _ = build_network(input_node, is_training_t)
    saver = tf.train.Saver(tf.global_variables())
    for var in tf.global_variables():
        if var.name == "Layer_1_W:0":
            l1_w = var
    with tf.Session() as sess:
        saver.restore(sess, filepath)
        layer1_W = sess.run(l1_w)
        target = layer1_W[:, index]
        plt.imshow(np.reshape(target, (28,28)))
        plt.show()


if __name__ == "__main__":
    #accs, ces = learning()
    #acc_array = np.array(accs)
    #x = np.arange(acc_array.shape[0])
    #plt.plot(x, acc_array)
    #plt.show()
    #ces_array = np.array(ces)
    #plt.plot(x, ces_array)
    #plt.show()

    (ac, ce) = learning()
