import matplotlib.pyplot as plt
import numpy as np

def part2():

    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainData = np.array([x.flatten() for x in trainData])
    validData = np.array([x.flatten() for x in validData])
    testData = np.array([x.flatten() for x in testData])


    newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)

    #2.2
    weights1 = {
        'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()),
        'wf1': tf.get_variable('W1', shape=(14*14*32,784), initializer=tf.contrib.layers.xavier_initializer()),
        'wf2': tf.get_variable('W2', shape=(784,10), initializer=tf.contrib.layers.xavier_initializer())
    }
    biases1 = {
        'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
        'bf1': tf.get_variable('B1', shape=(784), initializer=tf.contrib.layers.xavier_initializer()),
        'bf2': tf.get_variable('B2', shape=(10), initializer=tf.contrib.layers.xavier_initializer())
    }
    loss_train, accuracy_train, loss_valid, accuracy_valid, loss_test, accuracy_test = train_tensorflow(weights1, biases1, trainData, newtrain, validData, newvalid, testData, newtest)
    print('')

    plt.figure()
    plt.suptitle('cnn loss curves')
    plt.plot(loss_train,'',loss_valid,'',loss_test,'')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.grid()
    plt.legend(['Training Data Loss', 'Validation Data Loss', 'Test Data Loss'])
    plt.savefig('cnn_loss.png')


    plt.figure()
    plt.suptitle('cnn accuracy curves')
    plt.plot(accuracy_train,'',accuracy_valid,'',accuracy_test,'')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['Training Accuracy', 'Validation Accuracy', 'Test Accuracy'])
    plt.savefig('cnn_accuracy.png')


    #2.3.1
    weights2 = {
        'wc1': tf.get_variable('W10', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()),
        'wf1': tf.get_variable('W11', shape=(14*14*32,784), initializer=tf.contrib.layers.xavier_initializer()),
        'wf2': tf.get_variable('W12', shape=(784,10), initializer=tf.contrib.layers.xavier_initializer())
    }
    biases2 = {
        'bc1': tf.get_variable('B10', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
        'bf1': tf.get_variable('B11', shape=(784), initializer=tf.contrib.layers.xavier_initializer()),
        'bf2': tf.get_variable('B12', shape=(10), initializer=tf.contrib.layers.xavier_initializer())
    }
    _, accuracy_train1, _, accuracy_valid1, _, accuracy_test1 = train_tensorflow(weights2, biases2, trainData, newtrain, validData, newvalid, testData, newtest, 0.01)
    print('')
    weights3 = {
        'wc1': tf.get_variable('W20', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()),
        'wf1': tf.get_variable('W21', shape=(14*14*32,784), initializer=tf.contrib.layers.xavier_initializer()),
        'wf2': tf.get_variable('W22', shape=(784,10), initializer=tf.contrib.layers.xavier_initializer())
    }
    biases3 = {
        'bc1': tf.get_variable('B20', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
        'bf1': tf.get_variable('B21', shape=(784), initializer=tf.contrib.layers.xavier_initializer()),
        'bf2': tf.get_variable('B22', shape=(10), initializer=tf.contrib.layers.xavier_initializer())
    }
    _, accuracy_train2, _, accuracy_valid2, _, accuracy_test2 = train_tensorflow(weights3, biases3, trainData, newtrain, validData, newvalid, testData, newtest, 0.1)
    print('')
    weights4 = {
        'wc1': tf.get_variable('W30', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()),
        'wf1': tf.get_variable('W31', shape=(14*14*32,784), initializer=tf.contrib.layers.xavier_initializer()),
        'wf2': tf.get_variable('W32', shape=(784,10), initializer=tf.contrib.layers.xavier_initializer())
    }
    biases4 = {
        'bc1': tf.get_variable('B30', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
        'bf1': tf.get_variable('B31', shape=(784), initializer=tf.contrib.layers.xavier_initializer()),
        'bf2': tf.get_variable('B32', shape=(10), initializer=tf.contrib.layers.xavier_initializer())
    }
    _, accuracy_train3, _, accuracy_valid3, _, accuracy_test3 = train_tensorflow(weights4, biases4, trainData, newtrain, validData, newvalid, testData, newtest, 0.5)
    print('')

    print("\u03BB = 0.01, Final Accuracy of training: {}, validation: {}, testing: {}".format(accuracy_train1[-1], accuracy_valid1[-1], accuracy_test1[-1]))
    print("\u03BB = 0.1, Final Accuracy of training: {}, validation: {}, testing: {}".format(accuracy_train2[-1], accuracy_valid2[-1], accuracy_test2[-1]))
    print("\u03BB = 0.5 Final Accuracy of training: {}, validation: {}, testing: {}".format(accuracy_train3[-1], accuracy_valid3[-1], accuracy_test3[-1]))

    #2.3.2
    weights5 = {
        'wc1': tf.get_variable('W40', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()),
        'wf1': tf.get_variable('W41', shape=(14*14*32,784), initializer=tf.contrib.layers.xavier_initializer()),
        'wf2': tf.get_variable('W42', shape=(784,10), initializer=tf.contrib.layers.xavier_initializer())
    }
    biases5 = {
        'bc1': tf.get_variable('B40', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
        'bf1': tf.get_variable('B41', shape=(784), initializer=tf.contrib.layers.xavier_initializer()),
        'bf2': tf.get_variable('B42', shape=(10), initializer=tf.contrib.layers.xavier_initializer())
    }
    _, accuracy_train4, _, accuracy_valid4, _, accuracy_test4 = train_tensorflow(weights5, biases5, trainData, newtrain, validData, newvalid, testData, newtest, 0, 0.9)
    print('')
    weights6 = {
        'wc1': tf.get_variable('W50', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()),
        'wf1': tf.get_variable('W51', shape=(14*14*32,784), initializer=tf.contrib.layers.xavier_initializer()),
        'wf2': tf.get_variable('W52', shape=(784,10), initializer=tf.contrib.layers.xavier_initializer())
    }
    biases6 = {
        'bc1': tf.get_variable('B50', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
        'bf1': tf.get_variable('B51', shape=(784), initializer=tf.contrib.layers.xavier_initializer()),
        'bf2': tf.get_variable('B52', shape=(10), initializer=tf.contrib.layers.xavier_initializer())
    }
    _, accuracy_train5, _, accuracy_valid5, _, accuracy_test5 = train_tensorflow(weights6, biases6, trainData, newtrain, validData, newvalid, testData, newtest, 0, 0.75)
    print('')
    weights7 = {
        'wc1': tf.get_variable('W60', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()),
        'wf1': tf.get_variable('W61', shape=(14*14*32,784), initializer=tf.contrib.layers.xavier_initializer()),
        'wf2': tf.get_variable('W62', shape=(784,10), initializer=tf.contrib.layers.xavier_initializer())
    }
    biases7 = {
        'bc1': tf.get_variable('B60', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
        'bf1': tf.get_variable('B61', shape=(784), initializer=tf.contrib.layers.xavier_initializer()),
        'bf2': tf.get_variable('B62', shape=(10), initializer=tf.contrib.layers.xavier_initializer())
    }
    _, accuracy_train6, _, accuracy_valid6, _, accuracy_test6 = train_tensorflow(weights7, biases7, trainData, newtrain, validData, newvalid, testData, newtest, 0, 0.5)
    print('')

    plt.figure()
    plt.suptitle('dropout accuracy curves, p=0.9')
    plt.plot(accuracy_train4,'',accuracy_valid4,'',accuracy_test4,'')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['Training Accuracy', 'Validation Accuracy', 'Test Accuracy'])
    plt.savefig('dropout_accuracy_0.9.png')

    plt.figure()
    plt.suptitle('dropout accuracy curves, p=0.75')
    plt.plot(accuracy_train5,'',accuracy_valid5,'',accuracy_test5,'')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['Training Accuracy', 'Validation Accuracy', 'Test Accuracy'])
    plt.savefig('dropout_accuracy_0.75.png')

    plt.figure()
    plt.suptitle('dropout accuracy curves, p=0.5')
    plt.plot(accuracy_train6,'',accuracy_valid6,'',accuracy_test6,'')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['Training Accuracy', 'Validation Accuracy', 'Test Accuracy'])
    plt.savefig('dropout_accuracy_0.5.png')
