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
    
    weight1, bias1, loss1 = grad_descent(init_weight, init_bias, trainData, trainTarget, alpha1, epochs, reg, error_tol)
    weight2, bias2, loss2 = grad_descent(init_weight, init_bias, trainData, trainTarget, alpha2, epochs, reg, error_tol)
    weight3, bias3, loss3 = grad_descent(init_weight, init_bias, trainData, trainTarget, alpha3, epochs, reg, error_tol)

    plt.plot(loss1,'',loss2,'',loss3,'')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.grid()
    plt.legend(['MSE: \u03B1 = {}, \u03BB = {}'.format(0.005, reg),'MSE: \u03B1 = {}, \u03BB = {}'.format(0.001, reg),'MSE: \u03B1 = {}, \u03BB = {}'.format(0.0001, reg)])
    plt.savefig('loss_LinReg.png')