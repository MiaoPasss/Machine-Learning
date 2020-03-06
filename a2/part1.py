import matplotlib.pyplot as plt
import numpy as np

def part1():

    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainData = np.array([x.flatten() for x in trainData])
    validData = np.array([x.flatten() for x in validData])
    testData = np.array([x.flatten() for x in testData])


    newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)

    #1.3

    train_record, valid_record, test_record = train(trainData, newtrain, validData, testData)
    print('')

    loss_train = loss_calculation(newtrain, train_record)
    loss_valid = loss_calculation(newvalid, valid_record)
    loss_test = loss_calculation(newtest, test_record)

    accuracy_train, _ = accuracy_calculation(trainTarget, train_record)
    accuracy_valid, early_stop = accuracy_calculation(validTarget, valid_record)
    accuracy_test, _ = accuracy_calculation(testTarget, test_record)

    plt.figure()
    plt.suptitle('Cross Entropy loss curves')
    plt.plot(loss_train,'',loss_valid,'',loss_test,'')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.grid()
    plt.legend(['Training Data Loss', 'Validation Data Loss', 'Test Data Loss'])
    plt.savefig('Cross_Entropy_loss.png')


    plt.figure()
    plt.suptitle('Cross Entropy accuracy curves')
    plt.plot(accuracy_train,'',accuracy_valid,'',accuracy_test,'')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['Training Accuracy', 'Validation Accuracy', 'Test Accuracy'])
    plt.savefig('Cross_Entropy_accuracy.png')

    #1.4.1

    _, _, test_record = train(trainData, newtrain, validData, testData, 100)
    print('')
    accuracy_test1, _ = accuracy_calculation(testTarget, test_record)
    print('Number of Hidden Units = 100, final test accuracy = {}'.format(accuracy_test1[-1]))

    _, _, test_record = train(trainData, newtrain, validData, testData, 500)
    print('')
    accuracy_test2, _ = accuracy_calculation(testTarget, test_record)
    print('Number of Hidden Units = 500, final test accuracy = {}'.format(accuracy_test2[-1]))

    _, _, test_record = train(trainData, newtrain, validData, testData, 2000)
    print('')
    accuracy_test3, _ = accuracy_calculation(testTarget, test_record)
    print('Number of Hidden Units = 2000, final test accuracy = {}'.format(accuracy_test3[-1]))


    #1.4.2
    print("Early Stopping at epoch: {}".format(early_stop))
    print("Training early stop accuracy: {}".format(accuracy_train[early_stop]))
    print("Validation early stop accuracy: {}".format(accuracy_valid[early_stop]))
    print("Testing early stop accuracy: {}".format(accuracy_test[early_stop]))

    plt.figure()
    plt.suptitle('Early Stopping accuracy curves')
    plt.plot(accuracy_train,'',accuracy_valid,'',accuracy_test,'')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.axvline(x=early_stop, ls='--')
    plt.grid()
    plt.legend(['Training Accuracy', 'Validation Accuracy', 'Test Accuracy'])
    plt.savefig('Early_Stopping_accuracy.png')
