import matplotlib.pyplot as plt
import numpy as np
import starter

def print_info(loss, train_a, valid_a, test_a, type, alpha, reg, comp_time):    
    if type is "MSE":
        print('Batch GD with \u03B1 = {}, \u03BB = {}, MSE = {}, training accuracy = {}, valid accuracy = {}, test accuracy = {}, '
            'computation time = {} ms'.format(alpha, reg, loss, train_a, valid_a, test_a, int(comp_time * 1000)))
    elif type is "CE":
        print('Batch GD with \u03BB = {}, \u03BB = {}, CE = {}, training accuracy = {}, valid accuracy = {}, test accuracy = {}, '
            'computation time = {} ms'.format(alpha, reg, loss, train_a, valid_a, test_a, int(comp_time * 1000)))

def linreg():

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
    weight_train1, bias_train1, loss_train1 = grad_descent(init_weight, init_bias, trainData, trainTarget, alpha1, epochs, reg, error_tol, "CE")
    end1 = time.time()
    
    loss_valid1 = crossEntropyLoss(weight_train1, bias_train1, validData, validTarget, reg)
    loss_test1 = crossEntropyLoss(weight_train1, bias_train1, testData, testTarget, reg)

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
    plt.suptitle('Training accuracy')
    plt.plot(accuracy_train1,'',accuracy_train2,'',accuracy_train3,'')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['Training Accuracy', 'Validation Accuray', 'Test Accuracy'])
    plt.savefig('Cross_Entropy_accuracy_LogReg.png')
    

    print_info(loss_train1[-1], accuracy_train1[-1], accuracy_valid1[-1], accuracy_test1[-1], "CE", alpha1, reg, end1 - start1)
    print_info(loss_train2[-1], accuracy_train2[-1], accuracy_valid2[-1], accuracy_test2[-1], "CE", alpha2, reg, end2 - start2)
    print_info(loss_train3[-1], accuracy_train3[-1], accuracy_valid3[-1], accuracy_test3[-1], "CE", alpha3, reg, end3 - start3)


    #2.3

    reg = 0
    
    start4 = time.time()
    weight_train2, bias_train2, loss_train2 = grad_descent(init_weight, init_bias, trainData, trainTarget, alpha, epochs, reg1, error_tol)
    end4 = time.time()

    loss_valid2 = crossEntropyLoss(weight_train2, bias_train2, validData, validTarget, reg)
    loss_test2 = crossEntropyLoss(weight_train2, bias_train2, testData, testTarget, reg)

    accuracy_train2 = accuracy_calculation(weight_train2, bias_train2, trainData, trainTarget)
    accuracy_valid2 = accuracy_calculation(weight_train2, bias_train2, validData, validTarget)
    accuracy_test2 = accuracy_calculation(weight_train2, bias_train2, testData, testTarget)

    plt.figure()
    plt.suptitle('Training losses')
    plt.plot(loss_train4,'',loss_train5,'',loss_train6,'')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.grid()
    plt.legend(['MSE: \u03B1 = {}, \u03BB = {}'.format(alpha, reg1),'MSE: \u03B1 = {}, \u03BB = {}'.format(alpha, reg2),'MSE: \u03B1 = {}, \u03BB = {}'.format(alpha, reg3)])
    plt.savefig('Regulation_adjustment_training_loss_LinReg.png')


    plt.figure()
    plt.suptitle('Training accuracy')
    plt.plot(accuracy_train4,'',accuracy_train5,'',accuracy_train6,'')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['MSE: \u03B1 = {}, \u03BB = {}'.format(alpha, reg1),'MSE: \u03B1 = {}, \u03BB = {}'.format(alpha, reg2),'MSE: \u03B1 = {}, \u03BB = {}'.format(alpha, reg3)])
    plt.savefig('Regulation_adjustment_training_accuracy_LinReg.png')


    plt.figure()
    plt.suptitle('Validation accuracy')
    plt.plot(accuracy_valid4,'',accuracy_valid5,'',accuracy_valid6,'')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['MSE: \u03B1 = {}, \u03BB = {}'.format(alpha, reg1),'MSE: \u03B1 = {}, \u03BB = {}'.format(alpha, reg2),'MSE: \u03B1 = {}, \u03BB = {}'.format(alpha, reg3)])
    plt.savefig('Regulation_adjustment_validation_accuracy_LinReg.png')


    plt.figure()
    plt.suptitle('Testing accuracy')
    plt.plot(accuracy_test4,'',accuracy_test5,'',accuracy_test6,'')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['MSE: \u03B1 = {}, \u03BB = {}'.format(alpha, reg1),'MSE: \u03B1 = {}, \u03BB = {}'.format(alpha, reg2),'MSE: \u03B1 = {}, \u03BB = {}'.format(alpha, reg3)])
    plt.savefig('Regulation_adjustment_testing_accuracy_LinReg.png')

    print_info(loss_train4[-1], accuracy_train4[-1], accuracy_valid4[-1], accuracy_test4[-1], "GD", alpha, reg1, end4 - start4)
    print_info(loss_train5[-1], accuracy_train5[-1], accuracy_valid5[-1], accuracy_test5[-1], "GD", alpha, reg2, end5 - start5)
    print_info(loss_train6[-1], accuracy_train6[-1], accuracy_valid6[-1], accuracy_test6[-1], "GD", alpha, reg3, end6 - start6)


    #1.5
    start = time.time()
    w_normal_train, b_normal_train = normal_equation(trainData, trainTarget, reg)
    end = time.time()
    start1 = time.time()
    w_normal_train1, b_normal_train1 = normal_equation(trainData, trainTarget, reg1)
    end1 = time.time()
    start2 = time.time()
    w_normal_train2, b_normal_train2 = normal_equation(trainData, trainTarget, reg2)
    end2 = time.time()
    start3 = time.time()
    w_normal_train3, b_normal_train3 = normal_equation(trainData, trainTarget, reg3)
    end3 = time.time()
    
    loss_normal = MSE(w_normal_train, b_normal_train, trainData, trainTarget, reg)
    loss_normal1 = MSE(w_normal_train1, b_normal_train1, trainData, trainTarget, reg1)
    loss_normal2 = MSE(w_normal_train2, b_normal_train2, trainData, trainTarget, reg2)
    loss_normal3 = MSE(w_normal_train3, b_normal_train3, trainData, trainTarget, reg3)
    
    w_normal_train = [w_normal_train]
    w_normal_train1 = [w_normal_train1]
    w_normal_train2 = [w_normal_train2]
    w_normal_train3 = [w_normal_train3]
    b_normal_train = [b_normal_train]
    b_normal_train1 = [b_normal_train1]
    b_normal_train2 = [b_normal_train2]
    b_normal_train3 = [b_normal_train3]

    normal_accuracy_train = accuracy_calculation(w_normal_train, b_normal_train, trainData, trainTarget)
    normal_accuracy_train1 = accuracy_calculation(w_normal_train1, b_normal_train1, trainData, trainTarget)
    normal_accuracy_train2 = accuracy_calculation(w_normal_train2, b_normal_train2, trainData, trainTarget)
    normal_accuracy_train3 = accuracy_calculation(w_normal_train3, b_normal_train3, trainData, trainTarget)

    normal_accuracy_valid = accuracy_calculation(w_normal_train, b_normal_train, validData, validTarget)
    normal_accuracy_valid1 = accuracy_calculation(w_normal_train1, b_normal_train1, validData, validTarget)
    normal_accuracy_valid2 = accuracy_calculation(w_normal_train2, b_normal_train2, validData, validTarget)
    normal_accuracy_valid3 = accuracy_calculation(w_normal_train3, b_normal_train3, validData, validTarget)

    normal_accuracy_test = accuracy_calculation(w_normal_train, b_normal_train, testData, testTarget)
    normal_accuracy_test1 = accuracy_calculation(w_normal_train1, b_normal_train1, testData, testTarget)
    normal_accuracy_test2 = accuracy_calculation(w_normal_train2, b_normal_train2, testData, testTarget)
    normal_accuracy_test3 = accuracy_calculation(w_normal_train3, b_normal_train3, testData, testTarget)

    print_info(loss_normal, normal_accuracy_train[0], normal_accuracy_valid[0], normal_accuracy_test[0], "normal", 0, reg, end - start)
    print_info(loss_normal1, normal_accuracy_train1[0], normal_accuracy_valid1[0], normal_accuracy_test1[0], "normal", 0, reg1, end1 - start1)
    print_info(loss_normal2, normal_accuracy_train2[0], normal_accuracy_valid2[0], normal_accuracy_test2[0], "normal", 0, reg2, end2 - start2)
    print_info(loss_normal3, normal_accuracy_train3[0], normal_accuracy_valid3[0], normal_accuracy_test3[0], "normal", 0, reg3, end3 - start3)