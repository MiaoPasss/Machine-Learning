import tensorflow as tf
import starter
import numpy as np
import matplotlib.pyplot as plt


def SGD():
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainData = np.array([x.flatten() for x in trainData])
    validData = np.array([x.flatten() for x in validData])
    testData = np.array([x.flatten() for x in testData])
    
    accuracy_train = []
    loss_train = []
    accuracy_valid = []
    loss_valid = []
    accuracy_test = []
    loss_test = []

    epochs = 700
    minibatch_size = 500
    iteration = ceil(3500 / minibatch_size)

    x, y, W, b, reg, loss, optimizer = buildGraph()
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    for _ in range(0, epochs):
        random_index = np.random.shuffle(np.arange(3500))
        shuffled_data = np.array([trainData[i] for i in random_index])
        shuffled_target = np.array([trainTarget[i] for i in random_index])

        for i in range(0, iteration):
            minibatch_data = shuffled_data[i*minibatch_size:(i+1)*minibatch_size]
            minibatch_target = shuffled_target[i*minibatch_size:(i+1)*minibatch_size]
            _, weight_train, bias_train = session.run([optimizer, W, b], feed_dict={x: minibatch_data, y: minibatch_target, reg: 0})
        accuracy_train.append(accuracy_calculation(weight_train, bias_train, trainData, trainTarget))
        loss_train.append(MSE(weight_train, bias_train, trainData, trainTarget))
        accuracy_valid.append(accuracy_calculation(weight_train, bias_train, validData, validTarget))
        loss_valid.append(MSE(weight_train, bias_train, validData, validTarget))
        accuracy_test.append(accuracy_calculation(weight_train, bias_train, testData, testTarget))
        loss_test.append(MSE(weight_train, bias_train, testData, testTarget))







