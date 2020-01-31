def normal_equation (x, y, reg):
    data_size = x.shape[0]
    dimension = x.shape[1]
    x = np.append(np.ones((data_size, 1)), x, axis = 1)

    I = np.identity(dimension + 1)
    I[:, 0] = 0

    w_normal = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x), x) + reg * I), np.transpose(x)), y)
    b_normal = w_normal[0]
    w_normal = np.delete(w_normal, 0).reshape(dimension, 1)
    
    return w_normal, b_normal