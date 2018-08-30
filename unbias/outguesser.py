import numpy as np
import pandas

from training_data import sigmoid, AGENT_CHOICES


class Outguesser:
    def predict_next_choice(self, history):
        prediction = self.predict(self.model_parameters, history)
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
    choice_histories = np.zeros((history_length, number_of_data_points))
    for i in range(0, history_length):
        choice_histories[i, :] = choice_history[history_length - i - 1: choice_history.shape[0] - i - 1]
    choice_outcomes = choice_history[history_length:]
    return choice_histories, choice_outcomes


def linear_choice_history_dependent_model(history_weights, choice_history):
    history_length = len(history_weights) - 1
    if choice_history.shape[0] <= history_length:
        return history_weights
    else:
        in_data, out_data = separate_choices_sequences_into_history_and_choice(choice_history[:, 0], history_length)
        ones_row = np.ones((1, choice_history.shape[0] - history_length))
        in_data = np.concatenate((ones_row, in_data), axis=0)
        return momentum_gradient_descent(history_weights, in_data, out_data)


def choice_history_reward_history_model(history_weights, choice_history):
    history_length = int((len(history_weights) - 1) / 2)
    if choice_history.shape[0] <= history_length:
        return history_weights
    else:
        reward_history = -1 * np.multiply(choice_history[:, 0], choice_history[:, 1])
        reward_choice_history = np.multiply(choice_history[:, 0], reward_history)
        in_choice_data, out_choice_data = separate_choices_sequences_into_history_and_choice(choice_history[:, 0],
                                                                                             history_length)
        in_reward_choice_data, out_reward_data = separate_choices_sequences_into_history_and_choice(
            reward_choice_history, history_length)
        ones_row = np.ones((1, choice_history.shape[0] - history_length))
        in_data = np.concatenate((ones_row, in_choice_data), axis=0)
        in_data = np.concatenate((in_data, in_reward_choice_data), axis=0)
        return momentum_gradient_descent(history_weights, in_data, out_choice_data)


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

    w = np.array(initial_weighting_vector, copy=True)  # initialize descent
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
    w = np.array(initial_weighting_vector, copy=True)  # initialize descent
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


def linear_choice_history_dependent_model_predictor(model_parameters, choice_history):
    history_length = (len(model_parameters) - 1)
    if choice_history.shape[0] <= history_length:
        prediction = np.random.choice(AGENT_CHOICES, 1)[0]
    else:
        history_data = np.concatenate(([1], np.flip(choice_history[1 - len(model_parameters):, 0], 0)))
        prediction = maximum_a_posteriori(model_parameters, history_data)
    return prediction


def choice_history_reward_history_model_predictor(model_parameters, choice_history):
    history_length = int((len(model_parameters) - 1) / 2)
    if choice_history.shape[0] <= history_length:
        prediction = np.random.choice(AGENT_CHOICES, 1)[0]
    else:
        agent_choice_data = np.flip(choice_history[-history_length:, 0], 0)
        outguesser_choice_data = np.flip(choice_history[-history_length:, 1], 0)
        reward_data = -1 * np.multiply(agent_choice_data, outguesser_choice_data)
        reward_choice_data = np.multiply(agent_choice_data, reward_data)
        history_data = np.concatenate(([1], agent_choice_data))
        history_data = np.concatenate((history_data, reward_choice_data))
        prediction = maximum_a_posteriori(model_parameters, history_data)
    return prediction


def regularized_momentum_gradient_descent(initial_weighting_vector, in_data, out_data, dw_min=1e-3, steps=1000,
                                          learning_rate=0.1, lamb=0.1):
    """

    :param initial_weighting_vector: a vector of the form [b, w] where b is the bias and w is history weighing
    :param in_data:
    :param out_data:
    :param dw_min: convergence criterion
    :param steps:
    :param learning_rate:
    :param lamb: regularizer
    :return:
    """

    x_pre = in_data
    x_target = out_data

    dw_prev = 1e4
    w = initial_weighting_vector  # initialize descent
    gamma = 0.5
    v = np.zeros((len(initial_weighting_vector),))
    for i in range(1, steps + 1):
        dw = np.dot(x_pre, ((x_target + 1) / 2 - sigmoid(w, x_pre))) + lamb * w
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
    return AGENT_CHOICES[0] if p >= 0.5 else AGENT_CHOICES[1]
