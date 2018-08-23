import numpy

from unbias.training_data import generate_stationary_agent_choices


def test_number_of_trials_are_correct():
    number_of_trials = 10
    w = numpy.array([0, -1])
    initial_choices = numpy.array([-1])
    choices = generate_stationary_agent_choices(number_of_trials, w, initial_choices)
    assert choices.shape == (number_of_trials,)


def test_can_handle_multiple_history_lengths():
    number_of_trials = 10
    w = numpy.array([0, -1, 0.3])
    initial_choices = numpy.array([-1, 1])
    choices = generate_stationary_agent_choices(number_of_trials, w, initial_choices)
    assert choices.shape == (number_of_trials,)
