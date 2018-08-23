import numpy as np


def sigmoid(w, x):
    return 1 / (1 + np.exp(-np.dot(w, x)))


def generate_stationary_agent_choices(number_of_trials, w, initial_choices):
    agent_choices = np.zeros((number_of_trials, ))
    history_length = w.shape[0]
    agent_choices[1:history_length] = initial_choices
    for i in range(number_of_trials - history_length):
        history = np.concatenate([np.array([1]), agent_choices[i:i + history_length - 1]])
        probability_for_one = sigmoid(w, history)
        agent_choices[i + history_length] = \
        np.random.choice(AGENT_CHOICES, 1, p=[probability_for_one, 1 - probability_for_one])[0]
    return agent_choices


AGENT_CHOICES = [1, -1]
