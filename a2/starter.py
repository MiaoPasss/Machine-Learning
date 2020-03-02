import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def relu(x):
    return np.array([max(a,0) for a in x])

def softmax(x):
    x_max = x.max()
    avg = sum([np.exp(a - x_max) for a in x])
    return np.array([np.exp(a - x_max)/avg for a in x])


def computeLayer(X, W, b):
    return np.add(np.matmul(W,X), b)

def CE(target, prediction):
    return np.matmul(target.flatten(),np.log(prediction.flatten())) / prediction.shape[0]

def gradCE(target, prediction):
    return -np.divide(target, prediction) / prediction.shape[0]

def train(trainData, validData, num_epochs=200, input_size=28*28, num_units=1000, alpha = 1e-5, gamma = 0.99, class_num = 10):
    weight_hidden = np.random.normal(loc=0,scale=np.sqrt(2/(input_size+num_units)),size=(input_size,num_units))
    weight_output = np.random.normal(loc=0,scale=np.sqrt(2/(num_units+class_num)),size=(num_units,class_num))
    bias_hidden = np.random.normal(loc=0,scale=np.sqrt(2/(input_size+num_units)),size=(1,num_units))
    bias_output = np.random.normal(loc=0,scale=np.sqrt(2/(num_units+class_num)),size=(1,class_num))
    nu_old_hidden = np.full(shape=(input_size,num_units),fill_value=1e-5)
    nu_old_output = np.full(shape=(num_units,class_num),fill_value=1e-5)
    nu_new_hidden = np.zeros(shape=(input_size,num_units))
    nu_new_output = np.zeros(shape=(num_units,class_num))

    for _ in range(1,num_epochs):
        output_hidden = relu(computeLayer(np.array(trainData), weight_hidden, bias_hidden))
        prediction = softmax(computeLayer(output_hidden, weight_output, bias_output))

        gradient_o = gradCE(np.array(trainTarget), prediction)
        w_o = np.matmul(output_hidden.T, gradient_o) / prediction.shape[0]
        b_o = np.average(gradient_o,axis=0)
        nu_new_output = gamma * nu_old_output + alpha * w_o
        nu_old_output = nu_new_output
        weight_output = weight_output - nu_new_output
        bias_output = bias_output - alpha * b_o
    
        gradient_h = np.matmul(gradient_o, weight_output.T)
        w_h = np.matmul(np.array(trainData).T, np.where(output_hidden < 0, 0, gradient_h)) / prediction.shape[0]
        b_h = np.average(np.where(output_hidden < 0, 0, gradient_h),axis=0)
        nu_new_hidden = gamma * nu_old_hidden + alpha * w_h
        nu_old_hidden = nu_new_hidden
        weight_hidden = weight_hidden - nu_new_hidden
        bias_hidden = bias_hidden - alpha * b_h
    
    return "🐫总好牛啊"

#🐫总好牛啊
