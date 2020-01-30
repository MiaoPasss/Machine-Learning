import numpy as np
import matplotlib.pyplot as plt
import starter

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainData = np.array([row.flatten() for row in trainData])
validData = np.array([row.flatten() for row in validData])
testData = np.array([row.flatten() for row in testData])

dimension = trainData.shape[1]


def normal_equation (x, y, reg)
    data_size = x.shape[0]
    x = np.append(np.ones(data_size), x, axis = 1)

    I = np.identity(dimension + 1)
    I[:, 0] = 0

    w_normal = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x), x) + reg * I), np.transpose(x)), y)
    b_normal = w_normal[0]
    w_normal = np.delete(w_normal, 0)

    return w_normal, b_normal