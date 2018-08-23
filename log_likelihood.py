# -*- coding: utf-8 -*-
"""
This script implements the simplest bayesian model to infer parameters for a binary choice task.
"""

import matplotlib.pyplot as plt
import numpy as np

from unbias.game import Game
from unbias.outguesser import simple_gradient_descent, Outguesser, maximum_a_posteriori
from unbias.training_data import sigmoid

dummy_agent = 0. * np.ones(2)
dummy_agent[1] = -0.8

p_init = 0.5

w_0 = np.zeros(2)  # initial prior predicts unbiased agent

outguesser = Outguesser(simple_gradient_descent, maximum_a_posteriori, w_0)

# learn and generate chain, history dependence is only one time step
T = 500

record_model_parameters = np.zeros((2, T))
record_model_parameters[:, 0] = w_0

game = Game(outguesser)

agent_choices = np.zeros(T)
outguesser_choices = np.zeros(T)

agent_choices[0] = np.random.rand() < p_init
outguesser_choices[0] = np.random.rand() < p_init

game.add_trial(agent_choices[0], outguesser_choices[0])

h = 1
for i in range(1, T):
    # outguesser makes his choice

    outguesser_choices[i] = game.get_outguesser_response()

    # agent makes his choice
    h = np.array([agent_choices[i - 1]])
    agent_choices[i] = maximum_a_posteriori(dummy_agent, h)

    game.add_trial(agent_choices[i], outguesser_choices[i])

    record_model_parameters[:, i] = game.outguesser.model_parameters

plt.plot(dummy_agent[0] * np.ones((T,)), label='true bias', color='blue')
plt.plot(np.linspace(1, T, T), record_model_parameters[0, :], label='estimated bias', color='blue', alpha=0.6)
plt.plot(dummy_agent[1] * np.ones((T,)), label='true history dependence', color='red')
plt.plot(np.linspace(1, T, T), record_model_parameters[1, :], label='estimated history dependence', color='red', alpha=0.6)
plt.legend()
plt.show()

