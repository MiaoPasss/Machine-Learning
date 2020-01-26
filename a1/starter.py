import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
    
    y_hat = np.dot(x, W) + b
    mse_loss = (np.linalg.norm(y_hat - y)) ** 2
    wd_loss = (np.linalg.norm(np.dot(W)) ** 2) * reg / 2

    return (mse_loss + wd_loss)
    
def gradMSE(W, b, x, y, reg):
    N = len(y)
    y_hat = np.dot(x, W) + b

    grad_weight = (1/N) * np.dot(np.transpose(x), y_hat - y) + reg * W
    grad_bias = (1/N) * np.sum(y_hat - y)

    return grad_weight, grad_bias

def grad_descent(W, b, x, y, alpha, epochs, reg, error_tol, lossType = "MSE"):
    weight_record = []
    bias_record = []
    loss_record = []

    print("Calculating gradient descent of",lossType, "with alpha = ", alpha, ", regularizer = ", reg)
    current_weight = W
    current_bias = b
    
    if lossType is "MSE":
        for _ in range(0, epochs):
            grad_w, grad_b = gradMSE(current_weight, current_bias, x, y, reg)
            updated_weight = current_weight - np.multiply(alpha, grad_w)
            if np.linalg.norm(updated_weight - current_weight) < error_tol):
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
            if np.linalg.norm(updated_weight - current_weight) < error_tol):
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
    total_loss = 1/N * (np.sum(-np.multiply(y, np.log(y_hat) - np.multiply((1 - y), np.log(y_hat))))) + reg/2 * (np.linalg.norm(W) ** 2)
    return total_loss

def gradCE(W, b, x, y, reg):
    N,n = x.shape
    y_hat = 1./(1 + np.exp(-(np.dot(x,W) + b)))
    grad_weight = 1/N * np.dot(x.T, y - y_hat) + reg * np.linalg.norm(W)
    grad_bias = 1/N * np.sum(y - y_hat)
    return grad_weight, grad_bias

def buildGraph(loss="MSE"):
	#Initialize weight and bias tensors
	tf.set_random_seed(421)

	if loss == "MSE":
	# Your implementation
	
	elif loss == "CE":
	#Your implementation here

