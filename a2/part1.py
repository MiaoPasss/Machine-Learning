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

    loss_train = loss_calculation(newtrain, train_record)
    loss_valid = loss_calculation(newvalid, valid_record)
    loss_test = loss_calculation(newtest, test_record)

    accuracy_train = accuracy_calculation(trainTarget, train_record)
    accuracy_valid = accuracy_calculation(validTarget, valid_record)
    accuracy_test = accuracy_calculation(testTarget, test_record)

    plt.figure()
    plt.suptitle('Cross Entropy loss curves')
    plt.plot(loss_train,'',loss_valid,'',loss_test,'')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.grid()
    plt.legend(['Training Data Loss', 'Validation Data Loss', 'Test Data Loss'])
    plt.savefig('Cross_Entropy_loss_LogReg.png')


    plt.figure()
    plt.suptitle('Cross Entropy accuracy curves')
    plt.plot(accuracy_train,'',accuracy_valid,'',accuracy_test,'')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['Training Accuracy', 'Validation Accuracy', 'Test Accuracy'])
    plt.savefig('Cross_Entropy_accuracy_LogReg.png')
