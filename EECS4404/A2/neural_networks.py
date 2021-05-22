# Name: Supriyo Ghosh
# Student ID: 215318728
# Assignment: 2

import tensorflow as tf
import numpy as np

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

# Q1.1.1 layer-wise building block
def create_new_layer(input_tensor, num_hidden_units):
    '''
        @param input_tensor - outputs of the previous layer in the neural network, without the bias term.
        @param num_hidden_units - number of hidden units to use for this new layer
    '''
    # Create the new layer weight matrix using Xavier initialization
    input_dim = int(input_tensor.shape[-1])
    initializer = tf.contrib.layers.xavier_initializer()
    W_shape = [input_dim, num_hidden_units]
    W = tf.get_variable("Layer1_W", initializer=initializer(W_shape), dtype=tf.float32)
    # todo: zero initializer?
    b = tf.get_variable("Layer1_b", shape=[1, num_hidden_units], dtype=tf.float32)

    # MatMul the extended input tensor by the new weight matrix and add the biases
    output_tensor = tf.matmul(input_tensor, W) + b

    # Return this operation
    return output_tensor

# Q1.1.2 learning
def learning():
    xTrain, yTrain, xValid, yValid, xTest, yTest = load_notmnist_data()

    with tf.Graph().as_default():
        num_hidden_units = 1000
        decay = 0
        B = 500
        learning_rates = [0.01, 0.005, 0.001]
        iters = 5000

        num_iters_per_epoch = len(xTrain)//B # number of iterations we have to do for one epoch
        print("Num epochs = ",iters/num_iters_per_epoch)

        # hyperparameters
        learning_rate = tf.placeholder(dtype=tf.float32, name="learning-rate")

        # Get Data
        xTrainTensor = tf.constant(xTrain, dtype=tf.float32, name="X-Training")
        yTrainTensor = tf.constant(yTrain, dtype=tf.float32, name="Y-Training")
        xTestTensor = tf.constant(xTest, dtype=tf.float32, name="X-Test")
        yTestTensor = tf.constant(yTest, dtype=tf.float32, name="Y-Test")
        xValidTensor = tf.constant(xValid, dtype=tf.float32, name="X-Validation")
        yValidTensor = tf.constant(yValid, dtype=tf.float32, name="Y-Validation")

        Xslice, yslice = tf.train.slice_input_producer([xTrainTensor, yTrainTensor], num_epochs=None)

        Xbatch, ybatch = tf.train.batch([Xslice, yslice], batch_size = B)

        with tf.variable_scope("default") as scope:
            # Create neural network layers for training
            trainb_batchOutput = create_new_layer(Xbatch, num_hidden_units)
            trainb_activatedOutput = tf.nn.relu(trainb_batchOutput)

            scope.reuse_variables()
            layer1_w = tf.get_variable("Layer1_W", shape=[784, num_hidden_units], dtype=tf.float32)
            layer1_b = tf.get_variable("Layer1_b", shape=[1, num_hidden_units], dtype=tf.float32)

            train_output = tf.matmul(xTrainTensor, layer1_w) + layer1_b
            train_activatedOutput = tf.nn.relu(train_output)

            valid_output = tf.matmul(xValidTensor, layer1_w) + layer1_b
            valid_activatedOutput = tf.nn.relu(valid_output)

            test_output = tf.matmul(xTestTensor, layer1_w) + layer1_b
            test_activatedOutput = tf.nn.relu(test_output)

            outputWeights_size = [int(trainb_activatedOutput.shape[-1]), 10] # We want a [1,10] tensor to get probabilities for each class
            outputWeights = tf.Variable(tf.contrib.layers.xavier_initializer()(outputWeights_size), name="Output_W")
            outputBias = tf.Variable(0, dtype=tf.float32, name="Output_Bias")

            trainb_y_pred = tf.sigmoid(tf.matmul(trainb_activatedOutput, outputWeights) + outputBias)
            train_y_pred = tf.sigmoid(tf.matmul(train_activatedOutput, outputWeights) + outputBias)
            valid_y_pred = tf.sigmoid(tf.matmul(valid_activatedOutput, outputWeights) + outputBias)
            test_y_pred = tf.sigmoid(tf.matmul(test_activatedOutput, outputWeights) + outputBias)
            trainb_softmaxLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=trainb_y_pred, labels=ybatch)) + decay * tf.nn.l2_loss(layer1_w)
            train_softmaxLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=train_y_pred, labels=yTrainTensor)) + decay * tf.nn.l2_loss(layer1_w)

            train_accuracy = tf.count_nonzero(tf.equal(tf.argmax(train_y_pred, 1), tf.argmax(yTrainTensor, 1))) / yTrainTensor.shape[0]
            valid_accuracy = tf.count_nonzero(tf.equal(tf.argmax(valid_y_pred, 1), tf.argmax(yValidTensor, 1))) / yValidTensor.shape[0]
            test_accuracy = tf.count_nonzero(tf.equal(tf.argmax(test_y_pred, 1), tf.argmax(yTestTensor, 1))) / yTestTensor.shape[0]

        # optimizer function
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(trainb_softmaxLoss)

        # TODO run a lot of iterations, plot loss vs epochs and classification error vs epochs
        for r in learning_rates:
            loss_amounts = []
            train_accs = []
            test_accs = []
            valid_accs = []
            with tf.Session() as sess:
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                for i in range(iters):
                    sess.run([optimizer], feed_dict={learning_rate: r})
                    if (i % num_iters_per_epoch == 0):
                        t_loss, t_acc, v_acc, test_acc = sess.run([train_softmaxLoss, train_accuracy, valid_accuracy, test_accuracy])
                        print("Epoch: {}, Training Loss: {}, Accuracies: [{}, {}, {}]".format(i//num_iters_per_epoch, t_loss, t_acc, v_acc, test_acc))
                        loss_amounts.append(t_loss)
                        train_accs.append(t_acc)
                        test_accs.append(test_acc)
                        valid_accs.append(v_acc)
                np.save("1.1.2_r{}_loss".format(r), loss_amounts)
                np.save("1.1.2_r{}_train_acc".format(r), train_accs)
                np.save("1.1.2_r{}_test_acc".format(r), test_accs)
                np.save("1.1.2_r{}_valid_acc".format(r), valid_accs)
learning()