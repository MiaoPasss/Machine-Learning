import matplotlib.pyplot as plt
import numpy as np
import time

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainData = np.array([x.flatten() for x in trainData])
validData = np.array([x.flatten() for x in validData])
testData = np.array([x.flatten() for x in testData])

def linreg():
    #1.3

    alpha1, alpha2, alpha3 = 0.005, 0.001, 0.0001
    reg = 0
    epochs = 5000
    error_tol = 1e-7

    init_weight = np.random.normal(size=(784,1))
    init_bias = np.random.uniform(-1,1)

    weight_train1, bias_train1, loss_train1 = grad_descent(init_weight, init_bias, trainData, trainTarget, alpha1, epochs, reg, error_tol)
    weight_train2, bias_train2, loss_train2 = grad_descent(init_weight, init_bias, trainData, trainTarget, alpha2, epochs, reg, error_tol)
    weight_train3, bias_train3, loss_train3 = grad_descent(init_weight, init_bias, trainData, trainTarget, alpha3, epochs, reg, error_tol)

    accuracy_train1 = accuracy_calculation(weight_train1, bias_train1, trainData, trainTarget)
    accuracy_train2 = accuracy_calculation(weight_train2, bias_train2, trainData, trainTarget)
    accuracy_train3 = accuracy_calculation(weight_train3, bias_train3, trainData, trainTarget)

    plt.figure()
    plt.suptitle('Training losses')
    plt.plot(loss_train1,'',loss_train2,'',loss_train3,'')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.grid()
    plt.legend(['MSE: \u03B1 = {}, \u03BB = {}'.format(alpha1, reg),'MSE: \u03B1 = {}, \u03BB = {}'.format(alpha2, reg),'MSE: \u03B1 = {}, \u03BB = {}'.format(alpha3, reg)])
    plt.savefig('Learning_rate_adjustment_training_loss_LinReg.png')


    plt.figure()
    plt.suptitle('Training accuracy')
    plt.plot(accuracy_train1,'',accuracy_train2,'',accuracy_train3,'')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['MSE: \u03B1 = {}, \u03BB = {}'.format(alpha1, reg),'MSE: \u03B1 = {}, \u03BB = {}'.format(alpha2, reg),'MSE: \u03B1 = {}, \u03BB = {}'.format(alpha3, reg)])
    plt.savefig('Learning_rate_adjustment_training_accuracy_LinReg.png')

    accuracy_valid1 = accuracy_calculation(weight_train1, bias_train1, validData, validTarget)
    accuracy_valid2 = accuracy_calculation(weight_train2, bias_train2, validData, validTarget)
    accuracy_valid3 = accuracy_calculation(weight_train3, bias_train3, validData, validTarget)

    plt.figure()
    plt.suptitle('Validation accuracy')
    plt.plot(accuracy_valid1,'',accuracy_valid2,'',accuracy_valid3,'')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['MSE: \u03B1 = {}, \u03BB = {}'.format(alpha1, reg),'MSE: \u03B1 = {}, \u03BB = {}'.format(alpha2, reg),'MSE: \u03B1 = {}, \u03BB = {}'.format(alpha3, reg)])
    plt.savefig('Learning_rate_adjustment_validation_accuracy_LinReg.png')

    accuracy_test1 = accuracy_calculation(weight_train1, bias_train1, testData, testTarget)
    accuracy_test2 = accuracy_calculation(weight_train2, bias_train2, testData, testTarget)
    accuracy_test3 = accuracy_calculation(weight_train3, bias_train3, testData, testTarget)

    plt.figure()
    plt.suptitle('Testing accuracy')
    plt.plot(accuracy_test1,'',accuracy_test2,'',accuracy_test3,'')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['MSE: \u03B1 = {}, \u03BB = {}'.format(alpha1, reg),'MSE: \u03B1 = {}, \u03BB = {}'.format(alpha2, reg),'MSE: \u03B1 = {}, \u03BB = {}'.format(alpha3, reg)])
    plt.savefig('Learning_rate_adjustment_testing_accuracy_LinReg.png')


    #1.4
    alpha = 0.005
    reg1, reg2, reg3 = 0.001, 0.1, 0.5
    
    weight_train4, bias_train4, loss_train4 = grad_descent(init_weight, init_bias, trainData, trainTarget, alpha, epochs, reg1, error_tol)
    weight_train5, bias_train5, loss_train5 = grad_descent(init_weight, init_bias, trainData, trainTarget, alpha, epochs, reg2, error_tol)
    weight_train6, bias_train6, loss_train6 = grad_descent(init_weight, init_bias, trainData, trainTarget, alpha, epochs, reg3, error_tol)

    accuracy_train4 = accuracy_calculation(weight_train4, bias_train4, trainData, trainTarget)
    accuracy_train5 = accuracy_calculation(weight_train5, bias_train5, trainData, trainTarget)
    accuracy_train6 = accuracy_calculation(weight_train6, bias_train6, trainData, trainTarget)

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

    accuracy_valid4 = accuracy_calculation(weight_train4, bias_train4, validData, validTarget)
    accuracy_valid5 = accuracy_calculation(weight_train5, bias_train5, validData, validTarget)
    accuracy_valid6 = accuracy_calculation(weight_train6, bias_train6, validData, validTarget)

    plt.figure()
    plt.suptitle('Validation accuracy')
    plt.plot(accuracy_valid4,'',accuracy_valid5,'',accuracy_valid6,'')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['MSE: \u03B1 = {}, \u03BB = {}'.format(alpha, reg1),'MSE: \u03B1 = {}, \u03BB = {}'.format(alpha, reg2),'MSE: \u03B1 = {}, \u03BB = {}'.format(alpha, reg3)])
    plt.savefig('Regulation_adjustment_validation_accuracy_LinReg.png')

    accuracy_test4 = accuracy_calculation(weight_train4, bias_train4, testData, testTarget)
    accuracy_test5 = accuracy_calculation(weight_train5, bias_train5, testData, testTarget)
    accuracy_test6 = accuracy_calculation(weight_train6, bias_train6, testData, testTarget)

    plt.figure()
    plt.suptitle('Testing accuracy')
    plt.plot(accuracy_test4,'',accuracy_test5,'',accuracy_test6,'')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['MSE: \u03B1 = {}, \u03BB = {}'.format(alpha, reg1),'MSE: \u03B1 = {}, \u03BB = {}'.format(alpha, reg2),'MSE: \u03B1 = {}, \u03BB = {}'.format(alpha, reg3)])
    plt.savefig('Regulation_adjustment_testing_accuracy_LinReg.png')

