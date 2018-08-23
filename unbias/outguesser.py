import numpy as np

from training_data import sigmoid, AGENT_CHOICES


class Outguesser:
    def predict_next_choice(self):
        return self.predict(self.model_parameters)

    def update_model(self, data):
        self.model_parameters = self.optimize(self.model_parameters, data)

    def __init__(self, optimize, predict, model_parameters):
        """

        :param optimizer:
        :param model_parameters: vector of form [b,w]
        """
        self.model_parameters = model_parameters
        self.predict = predict
        self.optimize = optimize


def simple_gradient_descent(initial_weighting_vector, data, steps=100, learning_rate=0.001):
    """

    :param initial_weighting_vector: a vector of the form [b, w] where b is the bias and w is history weighing
    :param data: numpy array with N choices in {-1,1}
    :param steps:
    :param learning_rate:
    :return:
    """

    history_length = len(initial_weighting_vector) - 1
    number_of_data_points = data.shape[0] - history_length
    x_pre = np.ones((history_length + 1, number_of_data_points))
    for i in range(1, history_length + 1):
        x_pre[i, :] = data[i - 1:i + number_of_data_points - 1]

    x_target = data[history_length:]

    w = initial_weighting_vector  # initialize descent
    for i in range(1, steps):
        dw = np.dot(x_pre, (x_target - sigmoid(w, x_pre)))
        w += learning_rate * dw

    return w


def maximum_a_posteriori(model_parameters, history):
    p = sigmoid(model_parameters, history)
    return np.random.choice(AGENT_CHOICES, 1, p=[p, 1 - p])[0]
