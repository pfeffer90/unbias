# -*- coding: utf-8 -*-
"""
This script implements the simplest bayesian model to infer parameters for a binary choice task.
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(w,x):
    return 1/(1+np.exp(-np.dot(w,x)))


    
    
def optim(w_prev, S, h, x_curr):
    w = w_prev #initialize descent
    steps = 10000 #gradient descent steps
    e = .00001 #learning rate
    for i in range(1,steps):
        dw = (x_curr-sigmoid(w,h))*h
        w = w + e*dw
        
    return w

    
    
w_gen = 0.5*np.ones(2)
w_gen[1] = -0.5
w_0 = np.zeros(2) #initial prior predicts unbiased agent
p_init = sigmoid(w_gen,np.ones(2))

#learn and generate chain, history dependence is only one time step
T = 500
var = .1 #the variance of the diffusion 
w = np.zeros((2,T))
w[:,0] = w_0
x = np.zeros(T)
x[0] = np.random.rand() < p_init
h = np.ones(2)
L = np.zeros(T)
for i in range(1,T):
    #generate next input
    h[1] = x[i-1]
    p = sigmoid(w_gen,h)
    x[i] = np.random.rand() < p 

    #learn parameters
    w[:,i] = optim(w[:,i-1],var,h, x[i])
    if x[i] == 1:
        L[i] = -x[i]*np.log(sigmoid(w[:,i],h))
    else:
        L[i] = -(1-x[i])*np.log(1-sigmoid(w[:,i],h))

plt.plot(np.linspace(1,T,T),w[0,:])
plt.plot(np.linspace(1,T,T),w[1,:])
plt.show()

plt.plot(L)
plt.show()
