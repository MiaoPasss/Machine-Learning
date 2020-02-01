import numpy as np

def normal_equation (x, y, reg):
    data_size = x.shape[0]
    dimension = x.shape[1]
    x = np.append(np.ones((data_size, 1)), x, axis = 1)

    I = np.identity(dimension + 1)
    I[:, 0] = 0

    w_normal = (np.linalg.inv((x.T @ x)/data_size + reg * I) @ x.T/data_size) @ y
    b_normal = w_normal[0]
    w_normal = np.delete(w_normal, 0, 0)
    w_normal = w_normal.reshape(dimension, 1)
    
    return w_normal, b_normal