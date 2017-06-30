import numpy as np
import tensorflow as tf
from  matplotlib import pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Load data and get the shape
house_data = fetch_california_housing()
m, n = house_data.data.shape

# Scale Data
X_scaler = StandardScaler()
house_data.data = X_scaler.fit_transform(house_data.data)

data = np.c_[np.ones((m, 1)), house_data.data]

X = tf.constant(data, dtype=tf.float32, name="X")
y = tf.constant(house_data.target.reshape((-1, 1)), dtype=tf.float32, name="y")
XT = tf.transpose(X)

theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as s:
    theta_hat = theta.eval()
    plt.plot(theta_hat)
    plt.show()
    print(theta_hat)
