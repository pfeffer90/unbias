# -*- coding: utf-8 -*-
"""
This script implements the simplest bayesian model to infer parameters for a binary choice task.
"""

import matplotlib.pyplot as plt
import numpy as np

from unbias.outguesser import simple_gradient_descent
from unbias.training_data import sigmoid

w_gen = 0. * np.ones(2)
w_gen[1] = -0.8
w_0 = np.zeros(2)  # initial prior predicts unbiased agent
p_init = sigmoid(w_gen, np.ones(2))

# learn and generate chain, history dependence is only one time step
T = 500
nb_datapoints = 0
var = .05  # the variance of the diffusion
w = np.zeros((2, T))
w[:, 0] = w_0
x = np.zeros(T)
x[0] = np.random.rand() < p_init
h = np.ones(2)
L = np.zeros(T)
for i in range(1, T):
    # generate next input
    h[1] = x[i - 1]
    p = sigmoid(w_gen, h)
    x[i] = np.random.rand() < p

    # learn parameters
    w[:, i] = simple_gradient_descent(w[:, i - 1], x[:i + 1])
    if x[i] == 1:
        L[i] = -x[i] * np.log(sigmoid(w[:, i], h))
    else:
        L[i] = -(1 - x[i]) * np.log(1 - sigmoid(w[:, i], h))

plt.plot(w_gen[0] * np.ones((T,)), label='true bias', color='blue')
plt.plot(np.linspace(1, T, T), w[0, :], label='estimated bias', color='blue', alpha=0.6)
plt.plot(w_gen[1] * np.ones((T,)), label='true history dependence', color='red')
plt.plot(np.linspace(1, T, T), w[1, :], label='estimated history dependence', color='red', alpha=0.6)
plt.legend()
plt.show()

plt.plot(L)
plt.show()
