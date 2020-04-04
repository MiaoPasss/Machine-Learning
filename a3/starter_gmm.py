import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#import helper as hlp

# Loading data
#data = np.load('data100D.npy')
data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)

N = num_pts
D = dim

def MoG(k, is_valid=False):
  # For Validation set
  train_batch = N
  train_data = data
  if is_valid:
    valid_batch = int(num_pts / 3.0)
    train_batch = N - valid_batch
    np.random.seed(45689)
    rnd_idx = np.arange(num_pts)
    np.random.shuffle(rnd_idx)
    val_data = data[rnd_idx[:valid_batch]]
    train_data = data[rnd_idx[valid_batch:]]

  Mu = tf.Variable(tf.random.truncated_normal([k, D]))
  X = tf.placeholder(tf.float32, shape = [None, D])
  phi = tf.Variable(tf.random.truncated_normal([k, 1]))
  psi = tf.Variable(tf.random.truncated_normal([k, 1]))
  sigma = tf.exp(phi)
  log_pi = logsoftmax(psi)

  log_Gauss = log_GaussPDF(X,Mu,sigma)
  log_pstr = log_posterior(log_Gauss, log_pi)

  mle_predict = tf.argmax(log_pstr,axis=1)
  logloss = - tf.reduce_sum(reduce_logsumexp(log_Gauss + tf.transpose(log_pi)),axis=0)
  _, _, count = tf.unique_with_counts(mle_predict)
  percentage = tf.divide(count, train_batch)


  optimizer = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.99, epsilon=1e-5)
  optimizer = optimizer.minimize(logloss)

  train_record = []

  init = tf.global_variables_initializer()
  session = tf.InteractiveSession()
  session.run(init)

  for i in range(200):
    _, loss = session.run([optimizer, logloss], feed_dict = {X: train_data})
    train_record.append(loss)

  predict, percentages = session.run([mle_predict, percentage], feed_dict = {X: train_data})

  if is_valid is True:
    valid_loss = session.run(logloss, feed_dict = {X: val_data})
    session.close()
    return valid_loss

  else:
    plt.figure()
    plt.title('Loss with number of gaussian clusters = {}'.format(k))
    plt.plot(train_record,'')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['Training Loss'])
    plt.savefig('Gaussian_Training_Loss_k={}.png'.format(k))

    plt.figure()
    plt.title('Data Points with number of gaussian clusters = {}'.format(k))
    plt.scatter(data[:,0], data[:,1], c=predict, s=0.5, alpha=0.8)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid()
    plt.savefig('Gaussian_Data_Points_k={}.png'.format(k))

    print("Final value of phi and psi:")
    print(phi.eval())
    print(psi.eval())

    print("Final value of sigma and pi:")
    print(sigma.eval())
    print(tf.exp(log_pi).eval())


    session.close()
    return


# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    # TODO
    return tf.reduce_sum(tf.squared_difference(tf.expand_dims(X,1),tf.expand_dims(MU,0)), 2)

def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1

    # Outputs:
    # log Gaussian PDF N X K

    distance = distanceFunc(X,mu)
    exp = - tf.divide(distance, 2 * tf.transpose(sigma))
    coef = - (D / 2) * tf.log(2 * np.pi * tf.transpose(sigma))
    return tf.add(coef, exp)



def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    # TODO
    num = tf.add(log_PDF, tf.transpose(log_pi))
    den = reduce_logsumexp(num, keep_dims=True)
    return tf.subtract(num,den)

MoG(1, False)
MoG(2, False)
MoG(3, False)
MoG(4, False)
MoG(5, False)

print('k = 1, valid loss = {}'.format(MoG(1, True)))
print('k = 2, valid loss = {}'.format(MoG(2, True)))
print('k = 3, valid loss = {}'.format(MoG(3, True)))
print('k = 4, valid loss = {}'.format(MoG(4, True)))
print('k = 5, valid loss = {}'.format(MoG(5, True)))
