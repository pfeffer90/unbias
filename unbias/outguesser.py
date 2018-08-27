import numpy as np
import pandas

from training_data import sigmoid, AGENT_CHOICES


class Outguesser:
    def predict_next_choice(self, history):
        if history.shape[0] <= len(self.model_parameters) - 1:
            prediction = np.random.choice(AGENT_CHOICES, 1)[0]
        else:
            current_history = history[-len(self.model_parameters) + 1:]
            prediction = self.predict(self.model_parameters, current_history)
        return prediction

    def update_model(self, data):
        if data.shape[0] > len(self.model_parameters) - 1:
            self.model_parameters = self.optimize(self.model_parameters, data)
        if self.record:
            self.recording_data_frame = pandas.concat(
                [self.recording_data_frame, pandas.DataFrame(self._prepare_dict_with_model_params())],
                ignore_index=True)

    def __init__(self, optimize, predict, model_parameters, record=False):
        """

        :param optimizer:
        :param model_parameters: vector of form [b,w]
        """
        self.model_parameters = model_parameters
        self.predict = predict
        self.optimize = optimize
        self.record = record

        if self.record:
            self._initialize_history_data_frame()

    def _initialize_history_data_frame(self):
        self.recording_data_frame = pandas.DataFrame(self._prepare_dict_with_model_params())

    def _prepare_dict_with_model_params(self):
        data_dict_keys = ['b'] + ["w_-{:d}".format(i) for i in range(1, len(self.model_parameters))]
        model_dict = {}
        for key, value in zip(data_dict_keys, self.model_parameters):
            model_dict.update({key: [value]})
        return model_dict


def simple_gradient_descent(initial_weighting_vector, data, steps=100, learning_rate=0.05):
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
        dw = np.dot(x_pre, ((x_target + 1) / 2 - sigmoid(w, x_pre)))
        w += learning_rate * dw

    return w


def maximum_a_posteriori(model_parameters, history):
    p = sigmoid(model_parameters, np.concatenate((np.array([1]), history)))
    return np.random.choice(AGENT_CHOICES, 1, p=[p, 1 - p])[0]
