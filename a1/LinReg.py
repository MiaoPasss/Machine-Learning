import matplotlib.pyplot as plt
import numpy as np
import time

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainData = np.array([x.flatten() for x in trainData])
validData = np.array([x.flatten() for x in validData])
testData = np.array([x.flatten() for x in testData])

def timer(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        print (method.__name__, ': ', (te - ts) * 1000, 'ms')
        return result
    return timed

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
    plt.legend(['MSE: \u03B1 = {}, \u03BB = {}'.format(alpha1, reg),'MSE: \u03B1 = {}, \u03BB = {}'.format(alpha2, reg),'MSE: \u03B1 = {}, \u03BB = {}'.format(alpha3, reg)])
    plt.savefig('Learning_rate_adjustment_loss_LinReg.png')

    #1.4
    alpha = 0.005
    reg1, reg2, reg3 = 0.001, 0.1, 0.5
    epochs = 5000
    error_tol = 1e-7

    init_weight = np.random.normal(size=(784,1))
    init_bias = np.random.uniform(-1,1)
    
    weight1, bias1, loss1 = grad_descent(init_weight, init_bias, trainData, trainTarget, alpha, epochs, reg1, error_tol)
    weight2, bias2, loss2 = grad_descent(init_weight, init_bias, trainData, trainTarget, alpha, epochs, reg2, error_tol)
    weight3, bias3, loss3 = grad_descent(init_weight, init_bias, trainData, trainTarget, alpha, epochs, reg3, error_tol)

    plt.plot(loss1,'',loss2,'',loss3,'')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.grid()
    plt.legend(['MSE: \u03B1 = {}, \u03BB = {}'.format(alpha, reg1),'MSE: \u03B1 = {}, \u03BB = {}'.format(alpha, reg2),'MSE: \u03B1 = {}, \u03BB = {}'.format(alpha, reg3)])
    plt.savefig('Regulation_adjustment_loss_LinReg.png')