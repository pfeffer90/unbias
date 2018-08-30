# -*- coding: utf-8 -*-
"""
This script implements the simplest bayesian model to infer parameters for a binary choice task.
"""

import matplotlib.pyplot as plt
import numpy as np

from unbias.game import Game
from unbias.outguesser import Outguesser, choice_history_reward_history_model
from unbias.training_data import AGENT_CHOICES

p_init = 0.5


def random_predictor(_1, _2):
    return np.random.choice(AGENT_CHOICES, p=[0.5, 0.5])


weight_prior = np.zeros((3,))
outguesser = Outguesser(choice_history_reward_history_model, random_predictor, weight_prior)

T = 300

record_model_parameters = np.zeros((3, T))
record_model_parameters[:, 0] = weight_prior

game = Game(outguesser)

agent_choices = np.zeros(T)
outguesser_choices = np.zeros(T)

agent_choices[0] = 1
outguesser_choices[0] = -1

game.add_trial(agent_choices[0], outguesser_choices[0])

for i in range(1, T):
    h = np.array([agent_choices[i - 1]])

    # outguesser makes his choice
    outguesser_choices[i] = game.get_outguesser_response()

    # agent makes his choice, he repeats his action if he wins and switches otherwise
    # agent_choices[i] = agent_choices[i - 1] if outguesser_choices[i - 1] != agent_choices[i - 1] else -1 * \
    agent_choices[i] = -1 * agent_choices[i - 1] if outguesser_choices[i - 1] != agent_choices[i - 1] else \
    agent_choices[
        i - 1]

    game.add_trial(agent_choices[i], outguesser_choices[i])

    record_model_parameters[:, i] = game.outguesser.model_parameters

plt.plot(record_model_parameters[0, :], label='estimated bias', color='black', alpha=0.6)
plt.plot(record_model_parameters[1, :], label='estimated choice dependence', color='blue', alpha=0.6)
plt.plot(record_model_parameters[2, :], label='estimated reward dependence', color='red', alpha=0.6)
plt.legend()
plt.show()

# agent_score = np.cumsum(agent_choices != outguesser_choices)
# outguesser_score = np.cumsum(agent_choices == outguesser_choices)
# plt.plot(agent_score, label="Agent score")
# plt.plot(outguesser_score, label='Outguesser score')
# plt.legend()
# plt.show()
