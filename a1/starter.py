import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#def timer(method):
#    def timed(*args, **kwargs):
#        ts = time.time()
#        result = method(*args, **kwargs)
#        te = time.time()
#        print (method.__name__, ':', (te - ts) * 1000, 'ms')
#        return result
#    return timed

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def MSE(W, b, x, y, reg):
    y_hat = np.matmul(x, W) + b
    mse_loss = ((np.linalg.norm(y_hat - y)) ** 2)/x.shape[0]
    wd_loss = (np.linalg.norm(W) ** 2) * reg / 2

    return (mse_loss + wd_loss)
    
def gradMSE(W, b, x, y, reg):
    N = len(y)
    y_hat = np.matmul(x, W) + b
    grad_weight = (2/N) * np.dot(np.transpose(x), (y_hat - y)) + reg * W
    grad_bias = (1/N) * np.sum(y_hat - y)

    return grad_weight, grad_bias

#@timer
def grad_descent(W, b, x, y, alpha, epochs, reg, error_tol, lossType = "MSE"):
    weight_record = []
    bias_record = []
    loss_record = []

    current_weight = W.reshape(x.shape[1], 1)
    current_bias = b
    
    if lossType is "MSE":
        for _ in range(0, epochs):
            grad_w, grad_b = gradMSE(current_weight, current_bias, x, y, reg)
            updated_weight = current_weight - np.multiply(alpha, grad_w)
            if (np.linalg.norm(updated_weight - current_weight) < error_tol):
                break
            updated_bias = current_bias - np.multiply(alpha, grad_b)
            loss = MSE(updated_weight, updated_bias, x, y, reg)
            current_weight = updated_weight
            current_bias = updated_bias
            weight_record.append(updated_weight)
            bias_record.append(updated_bias)
            loss_record.append(loss)
    elif lossType is "CE":
        for _ in range(0, epochs):
            grad_w, grad_b = gradCE(current_weight, current_bias, x, y, reg)
            updated_weight = current_weight - np.multiply(alpha, grad_w)
            if (np.linalg.norm(updated_weight - current_weight) < error_tol):
                break
            updated_bias = current_bias - np.multiply(alpha, grad_b)
            loss = crossEntropyLoss(updated_weight, updated_bias, x, y, reg)
            current_weight = updated_weight
            current_bias = updated_bias
            weight_record.append(updated_weight)
            bias_record.append(updated_bias)
            loss_record.append(loss)

    else:
        exit -1
    return weight_record, bias_record, loss_record

    
def crossEntropyLoss(W, b, x, y, reg):
    N,n = x.shape
    y_hat = 1./(1 + np.exp(-(np.dot(x,W) + b)))
    total_loss = 1/N * (np.sum(-np.multiply(y, np.log(y_hat)) - np.multiply((1 - y), np.log(1 - y_hat)))) + reg/2 * (np.linalg.norm(W) ** 2)
    return total_loss

def gradCE(W, b, x, y, reg):
    N,n = x.shape
    y_hat = 1./(1 + np.exp(-(np.dot(x,W) + b)))
    grad_weight = 1/N * np.dot(x.T, y - y_hat) + reg * W
    grad_bias = 1/N * np.sum(y - y_hat)
    return grad_weight, grad_bias

def accuracy_calculation(W, b, x, y):
    acc = [np.sum((np.dot(x, W[i]) + b[i] >= 0.5) == y) / y.shape[0] for i in range(len(W))]
    return acc
    
def buildGraph(loss="MSE"):
    '''
    tf.random.truncated_normal(
        shape,
        mean=0.0,
        stddev=1.0,
        dtype=tf.dtypes.float32,
        seed=None,
        name=None
    )
    '''

    W = tf.Variable(tf.truncated_normal([784, 1], 0, 0.5))
    b = tf.Variable(0)

    x = tf.placeholder(tf.float32, shape = [None, 784])
    y = tf.placeholder(tf.float32, shape = [None, 1])
    reg = tf.placeholder(tf.float32)

    tf.set_random_seed(421)

    #tf.nn.l2_loss computes half of the norm without square root

    if loss == "MSE":
        y_hat = x @ W + b
        mse_loss = tf.reduce_mean(tf.square(y_hat - y))
        wd_loss = reg * tf.nn.l2_loss(W)
        loss = mse_loss + wd_loss

        optimizer = tf.train.AdamOptimizer(0.001)
        optimizer = optimizer.minimize(loss)
        return W, b, y_hat, y, loss, optimizer

    elif loss == "CE":
        y_hat = 1 / (1 + tf.math.exp(-(x @ W + b)))
        ce_loss = tf.reduce_mean(-y * tf.math.log(y_hat) - (1 - y) * tf.math.log(1 - y_hat))
        wd_loss = reg * tf.nn.l2_loss(W)
        loss = ce_loss + wd_loss

        optimizer = tf.train.AdamOptimizer(0.001)
        optimizer = optimizer.minimize(loss)
        return W, b, y_hat, y, loss, optimizer


def loss_calculation(W, b, x, y, reg, type="MSE"):
    loss_record = []

    if type is "MSE":
        for i in range(len(W)):
            loss = MSE(W[i], b[i], x, y, reg)
            loss_record.append(loss)

    elif type is "CE":
        for i in range(len(W)):
            loss = crossEntropyLoss(W[i], b[i], x, y, reg)
            loss_record.append(loss)

    return loss_record
