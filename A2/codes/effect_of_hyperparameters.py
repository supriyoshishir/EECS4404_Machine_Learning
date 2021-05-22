# Name: Supriyo Ghosh
# Student ID: 215318728
# Assignment: 2

import numpy as np
import tensorflow as tf

def load_notMNIST():
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

def create_new_layer(input_tensor, num_hidden_units):
    '''
        @param input_tensor - outputs of the previous layer in the neural network, without the bias term.
        @param num_hidden_units - number of hidden units to use for this new layer
    '''
    # Create the new layer weight matrix using Xavier initialization
    input_dim = int(input_tensor.shape[-1])
    initializer = tf.contrib.layers.xavier_initializer()
    W_shape = [input_dim, num_hidden_units]
    W = tf.get_variable("W", initializer=initializer(W_shape), dtype=tf.float32)
    # todo: zero initializer?
    b = tf.get_variable("b", shape=[1, num_hidden_units], dtype=tf.float32)

    # MatMul the extended input tensor by the new weight matrix and add the biases
    output_tensor = tf.matmul(input_tensor, W) + b

    # Return this operation
    return output_tensor

def number_of_hidden_units():
    # Constants
    B = 500
    iters = 5000
    learning_rates = [0.01, 0.005, 0.001]
    hidden_units = [100,500,1000]
    output_data = [[],[],[]]
    
    # Load data
    (trainData, trainTarget, validData, validTarget,
         testData, testTarget) = load_notMNIST()
    
    # Precalculations
    num_iters_per_epoch = len(trainData)//B # number of iterations we have to do for one epoch
    print("Num epochs = ",iters/num_iters_per_epoch)
    inds = np.arange(trainData.shape[0])
	
    # Set place-holders & variables
    X = tf.placeholder(tf.float32, shape=(None, trainData.shape[-1]), name='X')
    Y = tf.placeholder(tf.float32, shape=(None, 10), name='Y')
    learning_rate = tf.placeholder(tf.float32, name='learning-rate')
	
    for h in range(0, len(hidden_units)):
        for lr in range(len(learning_rates)):
            # Build graph
            with tf.variable_scope("layer1_"+str(hidden_units[h])+"_"+str(lr), reuse=tf.AUTO_REUSE):
                s_1 = create_new_layer(X, hidden_units[h])
            x_1 = tf.nn.relu(s_1)
            with tf.variable_scope("layer2_"+str(hidden_units[h])+"_"+str(lr), reuse=tf.AUTO_REUSE):
                s_2 = create_new_layer(x_1, 10)
            x_2 = tf.nn.softmax(s_2)
    		
            # Calculate loss & accuracy
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=s_2, labels=Y))
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(x_2, 1), tf.argmax(Y, 1)), tf.float32))
            
            print("Number of hidden units", hidden_units[h])
    
            with tf.Session() as sess:
                with tf.variable_scope("default", reuse=tf.AUTO_REUSE):
                    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    print("Learning rate = ",learning_rates[lr])
                    temp_output = []
                    for i in range(iters):
                        if (i % num_iters_per_epoch == 0):
                            np.random.shuffle(inds)
                        sess.run([optimizer], feed_dict={learning_rate: learning_rates[lr], 
                                 X: trainData[inds[B*(i%num_iters_per_epoch):B*((i+1)%num_iters_per_epoch)]], 
                                 Y: trainTarget[inds[B*(i%num_iters_per_epoch):B*((i+1)%num_iters_per_epoch)]]})
                        if (i % num_iters_per_epoch == 0):
                            t_loss, t_acc = sess.run([loss, accuracy], feed_dict={X: trainData, Y: trainTarget})
                            v_loss, v_acc = sess.run([loss, accuracy], feed_dict={X: validData, Y: validTarget})
                            test_loss, test_acc = sess.run([loss, accuracy], feed_dict={X: testData, Y: testTarget})
                            print("Epoch: {}, Training Loss: {}, Accuracies: [{}, {}, {}]".format(i//num_iters_per_epoch, t_loss, t_acc, v_acc, test_acc))
                            temp_output.append([t_loss, t_acc, v_acc, test_acc])
                    output_data[h].append(temp_output)
					
    np.save('Q1-2-1.npy', output_data)   
    return output_data

def number_of_layers():
    # Constants
    B = 250
    iters = 5000
    learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    hidden_units = [500]
    output_data = [[]]
    
    # Load data
    (trainData, trainTarget, validData, validTarget,
         testData, testTarget) = load_notMNIST()
    
    # Precalculations
    num_iters_per_epoch = len(trainData)//B # number of iterations we have to do for one epoch
    print("Num epochs = ",iters/num_iters_per_epoch)
    inds = np.arange(trainData.shape[0])
	
    # Set place-holders & variables
    X = tf.placeholder(tf.float32, shape=(None, trainData.shape[-1]), name='X')
    Y = tf.placeholder(tf.float32, shape=(None, 10), name='Y')
    learning_rate = tf.placeholder(tf.float32, name='learning-rate')
	
    for h in range(0, len(hidden_units)):
        for lr in range(len(learning_rates)):
            # Build graph
            with tf.variable_scope("layer1_"+str(hidden_units[h])+"_"+str(lr), reuse=tf.AUTO_REUSE):
                s_1 = create_new_layer(X, hidden_units[h])
            x_1 = tf.nn.relu(s_1)
            with tf.variable_scope("layer2_"+str(hidden_units[h])+"_"+str(lr), reuse=tf.AUTO_REUSE):
                s_2 = create_new_layer(x_1, hidden_units[h])
            x_2 = tf.nn.softmax(s_2)
            with tf.variable_scope("layer3_"+str(hidden_units[h])+"_"+str(lr), reuse=tf.AUTO_REUSE):
                s_3 = create_new_layer(x_2, 10)
            x_3 = tf.nn.softmax(s_3)
    		
            # Calculate loss & accuracy
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=s_3, labels=Y))
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(x_3, 1), tf.argmax(Y, 1)), tf.float32))
            
            print("Number of hidden layers: 2, Number of hidden units", hidden_units[h])
    
            with tf.Session() as sess:
                with tf.variable_scope("default", reuse=tf.AUTO_REUSE):
                    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    print("Learning rate = ",learning_rates[lr])
                    temp_output = []
                    for i in range(iters):
                        if (i % num_iters_per_epoch == 0):
                            np.random.shuffle(inds)
                        sess.run([optimizer], feed_dict={learning_rate: learning_rates[lr], 
                                 X: trainData[inds[B*(i%num_iters_per_epoch):B*((i+1)%num_iters_per_epoch)]], 
                                 Y: trainTarget[inds[B*(i%num_iters_per_epoch):B*((i+1)%num_iters_per_epoch)]]})
                        if (i % num_iters_per_epoch == 0):
                            t_loss, t_acc = sess.run([loss, accuracy], feed_dict={X: trainData, Y: trainTarget})
                            v_loss, v_acc = sess.run([loss, accuracy], feed_dict={X: validData, Y: validTarget})
                            test_loss, test_acc = sess.run([loss, accuracy], feed_dict={X: testData, Y: testTarget})
                            print("Epoch: {}, Training Loss: {}, Accuracies: [{}, {}, {}]".format(i//num_iters_per_epoch, t_loss, t_acc, v_acc, test_acc))
                            temp_output.append([t_loss, t_acc, v_acc, test_acc])
                    output_data[h].append(temp_output)
					
    np.save('Q1-2-2.npy', output_data)   
    return output_data	
	
#output = number_of_hidden_units()
output = number_of_layers()