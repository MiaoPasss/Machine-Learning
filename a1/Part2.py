import matplotlib.pyplot as plt
import numpy as np
import starter

def print_info(train_l, valid_l, test_l, train_a, valid_a, test_a, type, alpha, reg, comp_time):
    if type is "MSE":
        print('Batch GD with \u03B1 = {}, \u03BB = {}, training MSE = {}, validation MSE = {}, testing MSE = {}, '
            'training accuracy = {}, valid accuracy = {}, test accuracy = {}, '
            'computation time = {} ms'.format(alpha, reg, train_l, valid_l, test_l, train_a, valid_a, test_a, int(comp_time * 1000)))
    elif type is "CE":
        print('Batch GD with \u03B1 = {}, \u03BB = {}, training CE = {}, validation CE = {}, testing CE = {}, '
            'training accuracy = {}, valid accuracy = {}, test accuracy = {}, '
            'computation time = {} ms'.format(alpha, reg, train_l, valid_l, test_l, train_a, valid_a, test_a, int(comp_time * 1000)))

def logreg():

    import time

    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainData = np.array([x.flatten() for x in trainData])
    validData = np.array([x.flatten() for x in validData])
    testData = np.array([x.flatten() for x in testData])


    #2.2

    alpha = 0.005
    reg = 0.1
    epochs = 5000
    error_tol = 1e-7

    init_weight = np.random.normal(0, 0.5, size=(784,1))
    init_bias = 0

    start1 = time.time()
    weight_train1, bias_train1, loss_train1 = grad_descent(init_weight, init_bias, trainData, trainTarget, alpha, epochs, reg, error_tol, "CE")
    end1 = time.time()
    
    loss_valid1 = loss_calculation(weight_train1, bias_train1, validData, validTarget, reg, "CE")
    loss_test1 = loss_calculation(weight_train1, bias_train1, testData, testTarget, reg, "CE")

    accuracy_train1 = accuracy_calculation(weight_train1, bias_train1, trainData, trainTarget)
    accuracy_valid1 = accuracy_calculation(weight_train1, bias_train1, validData, validTarget)
    accuracy_test1 = accuracy_calculation(weight_train1, bias_train1, testData, testTarget)

    plt.figure()
    plt.suptitle('Cross Entropy loss curves')
    plt.plot(loss_train1,'',loss_valid1,'',loss_test1,'')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.grid()
    plt.legend(['Training Data Loss', 'Validation Data Loss', 'Test Data Loss'])
    plt.savefig('Cross_Entropy_loss_LogReg.png')


    plt.figure()
    plt.suptitle('Cross Entropy accuracy curves')
    plt.plot(accuracy_train1,'',accuracy_valid1,'',accuracy_test1,'')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['Training Accuracy', 'Validation Accuracy', 'Test Accuracy'])
    plt.savefig('Cross_Entropy_accuracy_LogReg.png')
    

    print_info(loss_train1[-1], loss_valid1[-1], loss_test1[-1], accuracy_train1[-1], accuracy_valid1[-1], accuracy_test1[-1], "CE", alpha, reg, end1 - start1)


    #2.3

    reg1 = 0
    
    weight_train2, bias_train2, loss_train2 = grad_descent(init_weight, init_bias, trainData, trainTarget, alpha, epochs, reg1, error_tol, "CE")
    weight_train3, bias_train3, loss_train3 = grad_descent(init_weight, init_bias, trainData, trainTarget, alpha, epochs, reg1, error_tol, "MSE")

    plt.figure()
    plt.suptitle('CE and MSE loss curves comparison')
    plt.plot(loss_train2,'',loss_train3,'')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.grid()
    plt.legend(['Cross Entropy', 'Mean Square Error'])
    plt.savefig('MSE_CE_loss_compare.png')