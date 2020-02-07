import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

def print_info(train_a, valid_a, test_a, type, batch_size, comp_time, beta1 = 0.9, beta2 = 0.999, eps = 1e-08):
    if type is "MSE":
        print('MSE SGD with minibatch size = {}, \u03B21 = {}, \u03B22 = {}, \u03B5 = {}, '
            'training accuracy = {}, valid accuracy = {}, test accuracy = {}, '
            'computation time = {} ms'.format(batch_size, beta1, beta2, eps, train_a, valid_a, test_a, int(comp_time * 1000)))
    elif type is "CE":
        print('CE SGD with minibatch size = {}, \u03B21 = {}, \u03B22 = {}, \u03B5 = {}, '
            'training accuracy = {}, valid accuracy = {}, test accuracy = {}, '
            'computation time = {} ms'.format(batch_size, beta1, beta2, eps, train_a, valid_a, test_a, int(comp_time * 1000)))



def SGD():
    
    import time

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

    accuracy_train1 = []
    loss_train1 = []
    accuracy_valid1 = []
    loss_valid1 = []
    accuracy_test1 = []
    loss_test1 = []

    accuracy_train2 = []
    loss_train2 = []
    accuracy_valid2 = []
    loss_valid2 = []
    accuracy_test2 = []
    loss_test2 = []

    accuracy_train3 = []
    loss_train3 = []
    accuracy_valid3 = []
    loss_valid3 = []
    accuracy_test3 = []
    loss_test3 = []




    # Adam optimizer initial learning rate = 0.001
    # lambda regularizer = 0
    epochs = 700


    # 3.2
    minibatch_size = 500
    iteration = math.ceil(3500 / minibatch_size)
    
    W, b, x, y, reg, y_hat, loss, optimizer = buildGraph()
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    start = time.time()
    for _ in range(0, epochs):
        random_index = np.arange(3500)
        np.random.shuffle(random_index)
        shuffled_data = np.array([trainData[i] for i in random_index])
        shuffled_target = np.array([trainTarget[i] for i in random_index])

        for i in range(0, iteration):
            minibatch_data = shuffled_data[i*minibatch_size:(i+1)*minibatch_size]
            minibatch_target = shuffled_target[i*minibatch_size:(i+1)*minibatch_size]
            _, weight_train, bias_train = session.run([optimizer, W, b], feed_dict={x: minibatch_data, y: minibatch_target, reg: 0})
        weight_train = [weight_train]
        bias_train = [bias_train]
        accuracy_train.append(accuracy_calculation(weight_train, bias_train, trainData, trainTarget)[0])
        loss_train.append(MSE(weight_train, bias_train, trainData, trainTarget, 0))
        accuracy_valid.append(accuracy_calculation(weight_train, bias_train, validData, validTarget)[0])
        loss_valid.append(MSE(weight_train, bias_train, validData, validTarget, 0))
        accuracy_test.append(accuracy_calculation(weight_train, bias_train, testData, testTarget)[0])
        loss_test.append(MSE(weight_train, bias_train, testData, testTarget, 0))
    end = time.time()

    session.close()


    plt.figure()
    plt.suptitle('Loss Curves: Minibatch SGD MSE with batch size = 500')
    plt.plot(loss_train,'',loss_valid,'',loss_test,'')
    plt.xlabel('epochs')
    plt.ylabel('MSE losses')
    plt.grid()
    plt.legend(['Training Data Loss', 'Validation Data Loss', 'Test Data Loss'])
    plt.savefig('SGD_MSE_loss_batch500.png')


    plt.figure()
    plt.suptitle('Accuracy Curves: Minibatch SGD MSE with batch size = 500')
    plt.plot(accuracy_train,'',accuracy_valid,'',accuracy_test,'')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['Training Accuracy', 'Validation Accuracy', 'Test Accuracy'])
    plt.savefig('SGD_MSE_accuracy_batch500.png')

    print_info(accuracy_train[-1], accuracy_valid[-1], accuracy_test[-1], "MSE", minibatch_size, end - start)

    # 3.3
    size1, size2, size3 = 100, 750, 1750
    iteration1 = math.ceil(3500 / size1)
    iteration2 = math.ceil(3500 / size2)
    iteration3 = math.ceil(3500 / size3)


    W, b, x, y, reg, y_hat, loss, optimizer = buildGraph()
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    start1 = time.time()
    for _ in range(0, epochs):
        random_index = np.arange(3500)
        np.random.shuffle(random_index)
        shuffled_data = np.array([trainData[i] for i in random_index])
        shuffled_target = np.array([trainTarget[i] for i in random_index])

        for i in range(0, iteration1):
            minibatch_data = shuffled_data[i*size1:(i+1)*size1]
            minibatch_target = shuffled_target[i*size1:(i+1)*size1]
            _, weight_train, bias_train = session.run([optimizer, W, b], feed_dict={x: minibatch_data, y: minibatch_target, reg: 0})
        loss_train1.append(MSE(weight_train, bias_train, trainData, trainTarget, 0))
        loss_valid1.append(MSE(weight_train, bias_train, validData, validTarget, 0))
        loss_test1.append(MSE(weight_train, bias_train, testData, testTarget, 0))
        weight_train = [weight_train]
        bias_train = [bias_train]
        accuracy_train1.append(accuracy_calculation(weight_train, bias_train, trainData, trainTarget)[0])
        accuracy_valid1.append(accuracy_calculation(weight_train, bias_train, validData, validTarget)[0])
        accuracy_test1.append(accuracy_calculation(weight_train, bias_train, testData, testTarget)[0])
    end1 = time.time()

    session.close()


    W, b, x, y, reg, y_hat, loss, optimizer = buildGraph()
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    start2 = time.time()
    for _ in range(0, epochs):
        random_index = np.arange(3500)
        np.random.shuffle(random_index)
        shuffled_data = np.array([trainData[i] for i in random_index])
        shuffled_target = np.array([trainTarget[i] for i in random_index])

        for i in range(0, iteration2):
            minibatch_data = shuffled_data[i*size2:(i+1)*size2]
            minibatch_target = shuffled_target[i*size2:(i+1)*size2]
            _, weight_train, bias_train = session.run([optimizer, W, b], feed_dict={x: minibatch_data, y: minibatch_target, reg: 0})
        loss_train2.append(MSE(weight_train, bias_train, trainData, trainTarget, 0))
        loss_valid2.append(MSE(weight_train, bias_train, validData, validTarget, 0))
        loss_test2.append(MSE(weight_train, bias_train, testData, testTarget, 0))
        weight_train = [weight_train]
        bias_train = [bias_train]
        accuracy_train2.append(accuracy_calculation(weight_train, bias_train, trainData, trainTarget)[0])
        accuracy_valid2.append(accuracy_calculation(weight_train, bias_train, validData, validTarget)[0])
        accuracy_test2.append(accuracy_calculation(weight_train, bias_train, testData, testTarget)[0])
    end2 = time.time()

    session.close()


    W, b, x, y, reg, y_hat, loss, optimizer = buildGraph()
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    start3 = time.time()
    for _ in range(0, epochs):
        random_index = np.arange(3500)
        np.random.shuffle(random_index)
        shuffled_data = np.array([trainData[i] for i in random_index])
        shuffled_target = np.array([trainTarget[i] for i in random_index])

        for i in range(0, iteration3):
            minibatch_data = shuffled_data[i*size3:(i+1)*size3]
            minibatch_target = shuffled_target[i*size3:(i+1)*size3]
            _, weight_train, bias_train = session.run([optimizer, W, b], feed_dict={x: minibatch_data, y: minibatch_target, reg: 0})
        loss_train3.append(MSE(weight_train, bias_train, trainData, trainTarget, 0))
        loss_valid3.append(MSE(weight_train, bias_train, validData, validTarget, 0))
        loss_test3.append(MSE(weight_train, bias_train, testData, testTarget, 0))
        weight_train = [weight_train]
        bias_train = [bias_train]
        accuracy_train3.append(accuracy_calculation(weight_train, bias_train, trainData, trainTarget)[0])
        accuracy_valid3.append(accuracy_calculation(weight_train, bias_train, validData, validTarget)[0])
        accuracy_test3.append(accuracy_calculation(weight_train, bias_train, testData, testTarget)[0])
    end3 = time.time()

    session.close()


    plt.figure()
    plt.suptitle('Loss Curves: Minibatch SGD MSE with batch size = 100')
    plt.plot(loss_train1,'',loss_valid1,'',loss_test1,'')
    plt.xlabel('epochs')
    plt.ylabel('MSE losses')
    plt.grid()
    plt.legend(['Training Data Loss', 'Validation Data Loss', 'Test Data Loss'])
    plt.savefig('SGD_MSE_loss_batch100.png')

    plt.figure()
    plt.suptitle('Accuracy Curves: Minibatch SGD MSE with batch size = 100')
    plt.plot(accuracy_train1,'',accuracy_valid1,'',accuracy_test1,'')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['Training Accuracy', 'Validation Accuracy', 'Test Accuracy'])
    plt.savefig('SGD_MSE_accuracy_batch100.png')


    plt.figure()
    plt.suptitle('Loss Curves: Minibatch SGD MSE with batch size = 750')
    plt.plot(loss_train2,'',loss_valid2,'',loss_test2,'')
    plt.xlabel('epochs')
    plt.ylabel('MSE losses')
    plt.grid()
    plt.legend(['Training Data Loss', 'Validation Data Loss', 'Test Data Loss'])
    plt.savefig('SGD_MSE_loss_batch750.png')

    plt.figure()
    plt.suptitle('Accuracy Curves: Minibatch SGD MSE with batch size = 750')
    plt.plot(accuracy_train2,'',accuracy_valid2,'',accuracy_test2,'')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['Training Accuracy', 'Validation Accuracy', 'Test Accuracy'])
    plt.savefig('SGD_MSE_accuracy_batch750.png')


    plt.figure()
    plt.suptitle('Loss Curves: Minibatch SGD MSE with batch size = 1750')
    plt.plot(loss_train3,'',loss_valid3,'',loss_test3,'')
    plt.xlabel('epochs')
    plt.ylabel('MSE losses')
    plt.grid()
    plt.legend(['Training Data Loss', 'Validation Data Loss', 'Test Data Loss'])
    plt.savefig('SGD_MSE_loss_batch1750.png')

    plt.figure()
    plt.suptitle('Accuracy Curves: Minibatch SGD MSE with batch size = 1750')
    plt.plot(accuracy_train3,'',accuracy_valid3,'',accuracy_test3,'')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['Training Accuracy', 'Validation Accuracy', 'Test Accuracy'])
    plt.savefig('SGD_MSE_accuracy_batch1750.png')

    print_info(accuracy_train1[-1], accuracy_valid1[-1], accuracy_test1[-1], "MSE", size1, end1 - start1)
    print_info(accuracy_train2[-1], accuracy_valid2[-1], accuracy_test2[-1], "MSE", size2, end2 - start2)
    print_info(accuracy_train3[-1], accuracy_valid3[-1], accuracy_test3[-1], "MSE", size3, end3 - start3)


    # 3.4

    accuracy_train4 = []
    accuracy_valid4 = []
    accuracy_test4 = []

    accuracy_train5 = []
    accuracy_valid5 = []
    accuracy_test5 = []

    accuracy_train6 = []
    accuracy_valid6 = []
    accuracy_test6 = []

    accuracy_train7 = []
    accuracy_valid7 = []
    accuracy_test7 = []

    accuracy_train8 = []
    accuracy_valid8 = []
    accuracy_test8 = []

    accuracy_train9 = []
    accuracy_valid9 = []
    accuracy_test9 = []

    W, b, x, y, reg, y_hat, loss, optimizer = buildGraph("MSE", 0.95)
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    start4 = time.time()
    for _ in range(0, epochs):
        random_index = np.arange(3500)
        np.random.shuffle(random_index)
        shuffled_data = np.array([trainData[i] for i in random_index])
        shuffled_target = np.array([trainTarget[i] for i in random_index])

        for i in range(0, iteration):
            minibatch_data = shuffled_data[i*minibatch_size:(i+1)*minibatch_size]
            minibatch_target = shuffled_target[i*minibatch_size:(i+1)*minibatch_size]
            _, weight_train, bias_train = session.run([optimizer, W, b], feed_dict={x: minibatch_data, y: minibatch_target, reg: 0})
        weight_train = [weight_train]
        bias_train = [bias_train]
        accuracy_train4.append(accuracy_calculation(weight_train, bias_train, trainData, trainTarget)[0])
        accuracy_valid4.append(accuracy_calculation(weight_train, bias_train, validData, validTarget)[0])
        accuracy_test4.append(accuracy_calculation(weight_train, bias_train, testData, testTarget)[0])
    end4 = time.time()

    session.close()


    W, b, x, y, reg, y_hat, loss, optimizer = buildGraph("MSE", 0.99)
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    start5 = time.time()
    for _ in range(0, epochs):
        random_index = np.arange(3500)
        np.random.shuffle(random_index)
        shuffled_data = np.array([trainData[i] for i in random_index])
        shuffled_target = np.array([trainTarget[i] for i in random_index])

        for i in range(0, iteration):
            minibatch_data = shuffled_data[i*minibatch_size:(i+1)*minibatch_size]
            minibatch_target = shuffled_target[i*minibatch_size:(i+1)*minibatch_size]
            _, weight_train, bias_train = session.run([optimizer, W, b], feed_dict={x: minibatch_data, y: minibatch_target, reg: 0})
        weight_train = [weight_train]
        bias_train = [bias_train]
        accuracy_train5.append(accuracy_calculation(weight_train, bias_train, trainData, trainTarget)[0])
        accuracy_valid5.append(accuracy_calculation(weight_train, bias_train, validData, validTarget)[0])
        accuracy_test5.append(accuracy_calculation(weight_train, bias_train, testData, testTarget)[0])
    end5 = time.time()

    session.close()


    W, b, x, y, reg, y_hat, loss, optimizer = buildGraph("MSE", 0.9, 0.99)
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    start6 = time.time()
    for _ in range(0, epochs):
        random_index = np.arange(3500)
        np.random.shuffle(random_index)
        shuffled_data = np.array([trainData[i] for i in random_index])
        shuffled_target = np.array([trainTarget[i] for i in random_index])

        for i in range(0, iteration):
            minibatch_data = shuffled_data[i*minibatch_size:(i+1)*minibatch_size]
            minibatch_target = shuffled_target[i*minibatch_size:(i+1)*minibatch_size]
            _, weight_train, bias_train = session.run([optimizer, W, b], feed_dict={x: minibatch_data, y: minibatch_target, reg: 0})
        weight_train = [weight_train]
        bias_train = [bias_train]
        accuracy_train6.append(accuracy_calculation(weight_train, bias_train, trainData, trainTarget)[0])
        accuracy_valid6.append(accuracy_calculation(weight_train, bias_train, validData, validTarget)[0])
        accuracy_test6.append(accuracy_calculation(weight_train, bias_train, testData, testTarget)[0])
    end6 = time.time()

    session.close()


    W, b, x, y, reg, y_hat, loss, optimizer = buildGraph("MSE", 0.9, 0.9999)
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    start7 = time.time()
    for _ in range(0, epochs):
        random_index = np.arange(3500)
        np.random.shuffle(random_index)
        shuffled_data = np.array([trainData[i] for i in random_index])
        shuffled_target = np.array([trainTarget[i] for i in random_index])

        for i in range(0, iteration):
            minibatch_data = shuffled_data[i*minibatch_size:(i+1)*minibatch_size]
            minibatch_target = shuffled_target[i*minibatch_size:(i+1)*minibatch_size]
            _, weight_train, bias_train = session.run([optimizer, W, b], feed_dict={x: minibatch_data, y: minibatch_target, reg: 0})
        weight_train = [weight_train]
        bias_train = [bias_train]
        accuracy_train7.append(accuracy_calculation(weight_train, bias_train, trainData, trainTarget)[0])
        accuracy_valid7.append(accuracy_calculation(weight_train, bias_train, validData, validTarget)[0])
        accuracy_test7.append(accuracy_calculation(weight_train, bias_train, testData, testTarget)[0])
    end7 = time.time()

    session.close()


    W, b, x, y, reg, y_hat, loss, optimizer = buildGraph("MSE", 0.9, 0.999, 1e-09)
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    start8 = time.time()
    for _ in range(0, epochs):
        random_index = np.arange(3500)
        np.random.shuffle(random_index)
        shuffled_data = np.array([trainData[i] for i in random_index])
        shuffled_target = np.array([trainTarget[i] for i in random_index])

        for i in range(0, iteration):
            minibatch_data = shuffled_data[i*minibatch_size:(i+1)*minibatch_size]
            minibatch_target = shuffled_target[i*minibatch_size:(i+1)*minibatch_size]
            _, weight_train, bias_train = session.run([optimizer, W, b], feed_dict={x: minibatch_data, y: minibatch_target, reg: 0})
        weight_train = [weight_train]
        bias_train = [bias_train]
        accuracy_train8.append(accuracy_calculation(weight_train, bias_train, trainData, trainTarget)[0])
        accuracy_valid8.append(accuracy_calculation(weight_train, bias_train, validData, validTarget)[0])
        accuracy_test8.append(accuracy_calculation(weight_train, bias_train, testData, testTarget)[0])
    end8 = time.time()

    session.close()


    W, b, x, y, reg, y_hat, loss, optimizer = buildGraph("MSE", 0.9, 0.999, 1e-04)
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    start9 = time.time()
    for _ in range(0, epochs):
        random_index = np.arange(3500)
        np.random.shuffle(random_index)
        shuffled_data = np.array([trainData[i] for i in random_index])
        shuffled_target = np.array([trainTarget[i] for i in random_index])

        for i in range(0, iteration):
            minibatch_data = shuffled_data[i*minibatch_size:(i+1)*minibatch_size]
            minibatch_target = shuffled_target[i*minibatch_size:(i+1)*minibatch_size]
            _, weight_train, bias_train = session.run([optimizer, W, b], feed_dict={x: minibatch_data, y: minibatch_target, reg: 0})
        weight_train = [weight_train]
        bias_train = [bias_train]
        accuracy_train9.append(accuracy_calculation(weight_train, bias_train, trainData, trainTarget)[0])
        accuracy_valid9.append(accuracy_calculation(weight_train, bias_train, validData, validTarget)[0])
        accuracy_test9.append(accuracy_calculation(weight_train, bias_train, testData, testTarget)[0])
    end9 = time.time()

    session.close()


    print_info(accuracy_train4[-1], accuracy_valid4[-1], accuracy_test4[-1], "MSE", minibatch_size, end4 - start4, 0.95)
    print_info(accuracy_train5[-1], accuracy_valid5[-1], accuracy_test5[-1], "MSE", minibatch_size, end5 - start5, 0.99)
    print_info(accuracy_train6[-1], accuracy_valid6[-1], accuracy_test6[-1], "MSE", minibatch_size, end6 - start6, 0.9, 0.99)
    print_info(accuracy_train7[-1], accuracy_valid7[-1], accuracy_test7[-1], "MSE", minibatch_size, end7 - start7, 0.9, 0.9999)
    print_info(accuracy_train8[-1], accuracy_valid8[-1], accuracy_test8[-1], "MSE", minibatch_size, end8 - start8, 0.9, 0.999, 1e-09)
    print_info(accuracy_train9[-1], accuracy_valid9[-1], accuracy_test9[-1], "MSE", minibatch_size, end9 - start9, 0.9, 0.999, 1e-04)




    # 3.5

    accuracy_train10 = []
    loss_train10 = []
    accuracy_valid10 = []
    loss_valid10 = []
    accuracy_test10 = []
    loss_test10 = []


    minibatch_size = 500
    iteration = math.ceil(3500 / minibatch_size)
    
    W, b, x, y, reg, y_hat, loss, optimizer = buildGraph("CE")
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    start10 = time.time()
    for _ in range(0, epochs):
        random_index = np.arange(3500)
        np.random.shuffle(random_index)
        shuffled_data = np.array([trainData[i] for i in random_index])
        shuffled_target = np.array([trainTarget[i] for i in random_index])

        for i in range(0, iteration):
            minibatch_data = shuffled_data[i*minibatch_size:(i+1)*minibatch_size]
            minibatch_target = shuffled_target[i*minibatch_size:(i+1)*minibatch_size]
            _, weight_train, bias_train = session.run([optimizer, W, b], feed_dict={x: minibatch_data, y: minibatch_target, reg: 0})
        loss_train10.append(crossEntropyLoss(weight_train, bias_train, trainData, trainTarget, 0))
        loss_valid10.append(crossEntropyLoss(weight_train, bias_train, validData, validTarget, 0))
        loss_test10.append(crossEntropyLoss(weight_train, bias_train, testData, testTarget, 0))
        weight_train = [weight_train]
        bias_train = [bias_train]
        accuracy_train10.append(accuracy_calculation(weight_train, bias_train, trainData, trainTarget)[0])     
        accuracy_valid10.append(accuracy_calculation(weight_train, bias_train, validData, validTarget)[0])
        accuracy_test10.append(accuracy_calculation(weight_train, bias_train, testData, testTarget)[0])

    end10 = time.time()

    session.close()


    plt.figure()
    plt.suptitle('Loss Curves: Minibatch SGD CE with batch size = 500')
    plt.plot(loss_train10,'',loss_valid10,'',loss_test10,'')
    plt.xlabel('epochs')
    plt.ylabel('CE losses')
    plt.grid()
    plt.legend(['Training Data Loss', 'Validation Data Loss', 'Test Data Loss'])
    plt.savefig('SGD_CE_loss_batch500.png')


    plt.figure()
    plt.suptitle('Accuracy Curves: Minibatch SGD CE with batch size = 500')
    plt.plot(accuracy_train10,'',accuracy_valid10,'',accuracy_test10,'')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['Training Accuracy', 'Validation Accuracy', 'Test Accuracy'])
    plt.savefig('SGD_CE_accuracy_batch500.png')

    print_info(accuracy_train10[-1], accuracy_valid10[-1], accuracy_test10[-1], "CE", minibatch_size, end10 - start10)


    accuracy_train11 = []
    loss_train11 = []
    accuracy_valid11 = []
    loss_valid11 = []
    accuracy_test11 = []
    loss_test11 = []

    accuracy_train12 = []
    loss_train12 = []
    accuracy_valid12 = []
    loss_valid12 = []
    accuracy_test12 = []
    loss_test12 = []

    accuracy_train13 = []
    loss_train13 = []
    accuracy_valid13 = []
    loss_valid13 = []
    accuracy_test13 = []
    loss_test13 = []


    size1, size2, size3 = 100, 750, 1750
    iteration1 = math.ceil(3500 / size1)
    iteration2 = math.ceil(3500 / size2)
    iteration3 = math.ceil(3500 / size3)


    W, b, x, y, reg, y_hat, loss, optimizer = buildGraph("CE")
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    start11 = time.time()
    for _ in range(0, epochs):
        random_index = np.arange(3500)
        np.random.shuffle(random_index)
        shuffled_data = np.array([trainData[i] for i in random_index])
        shuffled_target = np.array([trainTarget[i] for i in random_index])

        for i in range(0, iteration1):
            minibatch_data = shuffled_data[i*size1:(i+1)*size1]
            minibatch_target = shuffled_target[i*size1:(i+1)*size1]
            _, weight_train, bias_train = session.run([optimizer, W, b], feed_dict={x: minibatch_data, y: minibatch_target, reg: 0})
        loss_train11.append(crossEntropyLoss(weight_train, bias_train, trainData, trainTarget, 0))
        loss_valid11.append(crossEntropyLoss(weight_train, bias_train, validData, validTarget, 0))
        loss_test11.append(crossEntropyLoss(weight_train, bias_train, testData, testTarget, 0))
        weight_train = [weight_train]
        bias_train = [bias_train]
        accuracy_train11.append(accuracy_calculation(weight_train, bias_train, trainData, trainTarget)[0])
        accuracy_valid11.append(accuracy_calculation(weight_train, bias_train, validData, validTarget)[0])
        accuracy_test11.append(accuracy_calculation(weight_train, bias_train, testData, testTarget)[0])
    end11 = time.time()

    session.close()


    W, b, x, y, reg, y_hat, loss, optimizer = buildGraph("CE")
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    start12 = time.time()
    for _ in range(0, epochs):
        random_index = np.arange(3500)
        np.random.shuffle(random_index)
        shuffled_data = np.array([trainData[i] for i in random_index])
        shuffled_target = np.array([trainTarget[i] for i in random_index])

        for i in range(0, iteration2):
            minibatch_data = shuffled_data[i*size2:(i+1)*size2]
            minibatch_target = shuffled_target[i*size2:(i+1)*size2]
            _, weight_train, bias_train = session.run([optimizer, W, b], feed_dict={x: minibatch_data, y: minibatch_target, reg: 0})
        loss_train12.append(crossEntropyLoss(weight_train, bias_train, trainData, trainTarget, 0))
        loss_valid12.append(crossEntropyLoss(weight_train, bias_train, validData, validTarget, 0))
        loss_test12.append(crossEntropyLoss(weight_train, bias_train, testData, testTarget, 0))
        weight_train = [weight_train]
        bias_train = [bias_train]
        accuracy_train12.append(accuracy_calculation(weight_train, bias_train, trainData, trainTarget)[0])
        accuracy_valid12.append(accuracy_calculation(weight_train, bias_train, validData, validTarget)[0])
        accuracy_test12.append(accuracy_calculation(weight_train, bias_train, testData, testTarget)[0])
    end12 = time.time()

    session.close()


    W, b, x, y, reg, y_hat, loss, optimizer = buildGraph("CE")
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    start13 = time.time()
    for _ in range(0, epochs):
        random_index = np.arange(3500)
        np.random.shuffle(random_index)
        shuffled_data = np.array([trainData[i] for i in random_index])
        shuffled_target = np.array([trainTarget[i] for i in random_index])

        for i in range(0, iteration3):
            minibatch_data = shuffled_data[i*size3:(i+1)*size3]
            minibatch_target = shuffled_target[i*size3:(i+1)*size3]
            _, weight_train, bias_train = session.run([optimizer, W, b], feed_dict={x: minibatch_data, y: minibatch_target, reg: 0})
        loss_train13.append(crossEntropyLoss(weight_train, bias_train, trainData, trainTarget, 0))
        loss_valid13.append(crossEntropyLoss(weight_train, bias_train, validData, validTarget, 0))
        loss_test13.append(crossEntropyLoss(weight_train, bias_train, testData, testTarget, 0))
        weight_train = [weight_train]
        bias_train = [bias_train]
        accuracy_train13.append(accuracy_calculation(weight_train, bias_train, trainData, trainTarget)[0])
        accuracy_valid13.append(accuracy_calculation(weight_train, bias_train, validData, validTarget)[0])
        accuracy_test13.append(accuracy_calculation(weight_train, bias_train, testData, testTarget)[0])
    end13 = time.time()

    session.close()


    plt.figure()
    plt.suptitle('Loss Curves: Minibatch SGD CE with batch size = 100')
    plt.plot(loss_train11,'',loss_valid11,'',loss_test11,'')
    plt.xlabel('epochs')
    plt.ylabel('CE losses')
    plt.grid()
    plt.legend(['Training Data Loss', 'Validation Data Loss', 'Test Data Loss'])
    plt.savefig('SGD_CE_loss_batch100.png')

    plt.figure()
    plt.suptitle('Accuracy Curves: Minibatch SGD CE with batch size = 100')
    plt.plot(accuracy_train11,'',accuracy_valid11,'',accuracy_test11,'')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['Training Accuracy', 'Validation Accuracy', 'Test Accuracy'])
    plt.savefig('SGD_CE_accuracy_batch100.png')


    plt.figure()
    plt.suptitle('Loss Curves: Minibatch SGD CE with batch size = 750')
    plt.plot(loss_train12,'',loss_valid12,'',loss_test12,'')
    plt.xlabel('epochs')
    plt.ylabel('CE losses')
    plt.grid()
    plt.legend(['Training Data Loss', 'Validation Data Loss', 'Test Data Loss'])
    plt.savefig('SGD_CE_loss_batch750.png')

    plt.figure()
    plt.suptitle('Accuracy Curves: Minibatch SGD CE with batch size = 750')
    plt.plot(accuracy_train12,'',accuracy_valid12,'',accuracy_test12,'')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['Training Accuracy', 'Validation Accuracy', 'Test Accuracy'])
    plt.savefig('SGD_CE_accuracy_batch750.png')


    plt.figure()
    plt.suptitle('Loss Curves: Minibatch SGD CE with batch size = 1750')
    plt.plot(loss_train13,'',loss_valid13,'',loss_test13,'')
    plt.xlabel('epochs')
    plt.ylabel('CE losses')
    plt.grid()
    plt.legend(['Training Data Loss', 'Validation Data Loss', 'Test Data Loss'])
    plt.savefig('SGD_CE_loss_batch1750.png')

    plt.figure()
    plt.suptitle('Accuracy Curves: Minibatch SGD CE with batch size = 1750')
    plt.plot(accuracy_train13,'',accuracy_valid13,'',accuracy_test13,'')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['Training Accuracy', 'Validation Accuracy', 'Test Accuracy'])
    plt.savefig('SGD_CE_accuracy_batch1750.png')

    print_info(accuracy_train11[-1], accuracy_valid11[-1], accuracy_test11[-1], "CE", size1, end1 - start1)
    print_info(accuracy_train12[-1], accuracy_valid12[-1], accuracy_test12[-1], "CE", size2, end2 - start2)
    print_info(accuracy_train13[-1], accuracy_valid13[-1], accuracy_test13[-1], "CE", size3, end3 - start3)


    accuracy_train14 = []
    accuracy_valid14 = []
    accuracy_test14 = []

    accuracy_train15 = []
    accuracy_valid15 = []
    accuracy_test15 = []

    accuracy_train16 = []
    accuracy_valid16 = []
    accuracy_test16 = []

    accuracy_train17 = []
    accuracy_valid17 = []
    accuracy_test17 = []

    accuracy_train18 = []
    accuracy_valid18 = []
    accuracy_test18 = []

    accuracy_train19 = []
    accuracy_valid19 = []
    accuracy_test19 = []

    W, b, x, y, reg, y_hat, loss, optimizer = buildGraph("CE", 0.95)
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    start4 = time.time()
    for _ in range(0, epochs):
        random_index = np.arange(3500)
        np.random.shuffle(random_index)
        shuffled_data = np.array([trainData[i] for i in random_index])
        shuffled_target = np.array([trainTarget[i] for i in random_index])

        for i in range(0, iteration):
            minibatch_data = shuffled_data[i*minibatch_size:(i+1)*minibatch_size]
            minibatch_target = shuffled_target[i*minibatch_size:(i+1)*minibatch_size]
            _, weight_train, bias_train = session.run([optimizer, W, b], feed_dict={x: minibatch_data, y: minibatch_target, reg: 0})
        weight_train = [weight_train]
        bias_train = [bias_train]
        accuracy_train14.append(accuracy_calculation(weight_train, bias_train, trainData, trainTarget)[0])
        accuracy_valid14.append(accuracy_calculation(weight_train, bias_train, validData, validTarget)[0])
        accuracy_test14.append(accuracy_calculation(weight_train, bias_train, testData, testTarget)[0])
    end4 = time.time()

    session.close()


    W, b, x, y, reg, y_hat, loss, optimizer = buildGraph("CE", 0.99)
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    start5 = time.time()
    for _ in range(0, epochs):
        random_index = np.arange(3500)
        np.random.shuffle(random_index)
        shuffled_data = np.array([trainData[i] for i in random_index])
        shuffled_target = np.array([trainTarget[i] for i in random_index])

        for i in range(0, iteration):
            minibatch_data = shuffled_data[i*minibatch_size:(i+1)*minibatch_size]
            minibatch_target = shuffled_target[i*minibatch_size:(i+1)*minibatch_size]
            _, weight_train, bias_train = session.run([optimizer, W, b], feed_dict={x: minibatch_data, y: minibatch_target, reg: 0})
        weight_train = [weight_train]
        bias_train = [bias_train]
        accuracy_train15.append(accuracy_calculation(weight_train, bias_train, trainData, trainTarget)[0])
        accuracy_valid15.append(accuracy_calculation(weight_train, bias_train, validData, validTarget)[0])
        accuracy_test15.append(accuracy_calculation(weight_train, bias_train, testData, testTarget)[0])
    end5 = time.time()

    session.close()


    W, b, x, y, reg, y_hat, loss, optimizer = buildGraph("CE", 0.9, 0.99)
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    start6 = time.time()
    for _ in range(0, epochs):
        random_index = np.arange(3500)
        np.random.shuffle(random_index)
        shuffled_data = np.array([trainData[i] for i in random_index])
        shuffled_target = np.array([trainTarget[i] for i in random_index])

        for i in range(0, iteration):
            minibatch_data = shuffled_data[i*minibatch_size:(i+1)*minibatch_size]
            minibatch_target = shuffled_target[i*minibatch_size:(i+1)*minibatch_size]
            _, weight_train, bias_train = session.run([optimizer, W, b], feed_dict={x: minibatch_data, y: minibatch_target, reg: 0})
        weight_train = [weight_train]
        bias_train = [bias_train]
        accuracy_train16.append(accuracy_calculation(weight_train, bias_train, trainData, trainTarget)[0])
        accuracy_valid16.append(accuracy_calculation(weight_train, bias_train, validData, validTarget)[0])
        accuracy_test16.append(accuracy_calculation(weight_train, bias_train, testData, testTarget)[0])
    end6 = time.time()

    session.close()


    W, b, x, y, reg, y_hat, loss, optimizer = buildGraph("CE", 0.9, 0.9999)
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    start7 = time.time()
    for _ in range(0, epochs):
        random_index = np.arange(3500)
        np.random.shuffle(random_index)
        shuffled_data = np.array([trainData[i] for i in random_index])
        shuffled_target = np.array([trainTarget[i] for i in random_index])

        for i in range(0, iteration):
            minibatch_data = shuffled_data[i*minibatch_size:(i+1)*minibatch_size]
            minibatch_target = shuffled_target[i*minibatch_size:(i+1)*minibatch_size]
            _, weight_train, bias_train = session.run([optimizer, W, b], feed_dict={x: minibatch_data, y: minibatch_target, reg: 0})
        weight_train = [weight_train]
        bias_train = [bias_train]
        accuracy_train17.append(accuracy_calculation(weight_train, bias_train, trainData, trainTarget)[0])
        accuracy_valid17.append(accuracy_calculation(weight_train, bias_train, validData, validTarget)[0])
        accuracy_test17.append(accuracy_calculation(weight_train, bias_train, testData, testTarget)[0])
    end7 = time.time()

    session.close()


    W, b, x, y, reg, y_hat, loss, optimizer = buildGraph("CE", 0.9, 0.999, 1e-09)
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    start8 = time.time()
    for _ in range(0, epochs):
        random_index = np.arange(3500)
        np.random.shuffle(random_index)
        shuffled_data = np.array([trainData[i] for i in random_index])
        shuffled_target = np.array([trainTarget[i] for i in random_index])

        for i in range(0, iteration):
            minibatch_data = shuffled_data[i*minibatch_size:(i+1)*minibatch_size]
            minibatch_target = shuffled_target[i*minibatch_size:(i+1)*minibatch_size]
            _, weight_train, bias_train = session.run([optimizer, W, b], feed_dict={x: minibatch_data, y: minibatch_target, reg: 0})
        weight_train = [weight_train]
        bias_train = [bias_train]
        accuracy_train18.append(accuracy_calculation(weight_train, bias_train, trainData, trainTarget)[0])
        accuracy_valid18.append(accuracy_calculation(weight_train, bias_train, validData, validTarget)[0])
        accuracy_test18.append(accuracy_calculation(weight_train, bias_train, testData, testTarget)[0])
    end8 = time.time()

    session.close()


    W, b, x, y, reg, y_hat, loss, optimizer = buildGraph("CE", 0.9, 0.999, 1e-04)
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    start9 = time.time()
    for _ in range(0, epochs):
        random_index = np.arange(3500)
        np.random.shuffle(random_index)
        shuffled_data = np.array([trainData[i] for i in random_index])
        shuffled_target = np.array([trainTarget[i] for i in random_index])

        for i in range(0, iteration):
            minibatch_data = shuffled_data[i*minibatch_size:(i+1)*minibatch_size]
            minibatch_target = shuffled_target[i*minibatch_size:(i+1)*minibatch_size]
            _, weight_train, bias_train = session.run([optimizer, W, b], feed_dict={x: minibatch_data, y: minibatch_target, reg: 0})
        weight_train = [weight_train]
        bias_train = [bias_train]
        accuracy_train19.append(accuracy_calculation(weight_train, bias_train, trainData, trainTarget)[0])
        accuracy_valid19.append(accuracy_calculation(weight_train, bias_train, validData, validTarget)[0])
        accuracy_test19.append(accuracy_calculation(weight_train, bias_train, testData, testTarget)[0])
    end9 = time.time()

    session.close()


    print_info(accuracy_train14[-1], accuracy_valid14[-1], accuracy_test14[-1], "CE", minibatch_size, end4 - start4, 0.95)
    print_info(accuracy_train15[-1], accuracy_valid15[-1], accuracy_test15[-1], "CE", minibatch_size, end5 - start5, 0.99)
    print_info(accuracy_train16[-1], accuracy_valid16[-1], accuracy_test16[-1], "CE", minibatch_size, end6 - start6, 0.9, 0.99)
    print_info(accuracy_train17[-1], accuracy_valid17[-1], accuracy_test17[-1], "CE", minibatch_size, end7 - start7, 0.9, 0.9999)
    print_info(accuracy_train18[-1], accuracy_valid18[-1], accuracy_test18[-1], "CE", minibatch_size, end8 - start8, 0.9, 0.999, 1e-09)
    print_info(accuracy_train19[-1], accuracy_valid19[-1], accuracy_test19[-1], "CE", minibatch_size, end9 - start9, 0.9, 0.999, 1e-04)