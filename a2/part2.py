import matplotlib.pyplot as plt
import numpy as np

def part2():

    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainData = np.array([x.flatten() for x in trainData])
    validData = np.array([x.flatten() for x in validData])
    testData = np.array([x.flatten() for x in testData])


    newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)

    #2.2

    loss_train, accuracy_train = train_tensorflow(trainData, trainTarget)
    loss_valid, accuracy_valid = train_tensorflow(validData, validTarget)
    loss_test, accuracy_test = train_tensorflow(testData, testTarget)

    loss_train = loss_calculation(newtrain, train_record)
    loss_valid = loss_calculation(newvalid, valid_record)
    loss_test = loss_calculation(newtest, test_record)

    accuracy_train, _ = accuracy_calculation(trainTarget, train_record)
    accuracy_valid, _ = accuracy_calculation(validTarget, valid_record)
    accuracy_test, _ = accuracy_calculation(testTarget, test_record)

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
