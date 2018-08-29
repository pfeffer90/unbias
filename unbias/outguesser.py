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


def separate_choices_sequences_into_history_and_choice(choice_history, history_length):
    number_of_data_points = choice_history.shape[0] - history_length
    choice_histories = np.ones((history_length + 1, number_of_data_points))
    for i in range(1, history_length + 1):
        choice_histories[i, :] = choice_history[i - 1:i + number_of_data_points - 1]  # TODO change
    choice_outcomes = choice_history[history_length:]
    return choice_histories, choice_outcomes


def linear_choice_history_dependent_model(history_weights, choice_history):
    history_length = len(history_weights)-1
    if choice_history.shape[0] <= history_length:
        return history_weights
    else:
        in_data, out_data = separate_choices_sequences_into_history_and_choice(choice_history, history_length)
        return simple_gradient_descent(history_weights, in_data, out_data)


def simple_gradient_descent(initial_weighting_vector, in_data, out_data, steps=100, learning_rate=0.05):
    """

    :param initial_weighting_vector: a vector of the form [b, w] where b is the bias and w is history weighing
    :param data: numpy array with N choices in {-1,1}
    :param steps:
    :param learning_rate:
    :return:
    """

    x_pre = in_data
    x_target = out_data

    w = initial_weighting_vector  # initialize descent
    for i in range(1, steps):
        dw = np.dot(x_pre, ((x_target + 1) / 2 - sigmoid(w, x_pre)))
        w += learning_rate * dw
    return w


def momentum_gradient_descent(initial_weighting_vector, in_data, out_data, dw_min=1e-3, steps=1000, learning_rate=0.1):
    """

    :param initial_weighting_vector: a vector of the form [b, w] where b is the bias and w is history weighing
    :param in_data:
    :param out_data:
    :param dw_min: convergence criterion
    :param steps:
    :param learning_rate:
    :return:
    """

    x_pre = in_data
    x_target = out_data

    dw_prev = 1e4
    w = initial_weighting_vector  # initialize descent
    gamma = 0.5
    v = np.zeros((len(initial_weighting_vector),))
    for i in range(1, steps + 1):
        dw = np.dot(x_pre, ((x_target + 1) / 2 - sigmoid(w, x_pre)))
        v = gamma * v + learning_rate * dw
        w += v

        if np.linalg.norm(dw) < dw_min:
            break
        if np.linalg.norm(dw) > np.linalg.norm(dw_prev):
            learning_rate /= 2
        dw_prev = dw

    return w


def maximum_a_posteriori(model_parameters, history):
    p = sigmoid(model_parameters, history)
    return np.random.choice(AGENT_CHOICES, 1, p=[p, 1 - p])[0]
