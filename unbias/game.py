import json

import numpy as np
import pandas as pd

from interfaces import game_variants
from outguesser import Outguesser, choice_history_reward_history_model, choice_history_reward_history_model_predictor, \
    linear_choice_history_dependent_model, linear_choice_history_dependent_model_predictor
from store import GameMetaData, save_game


class GameConstants:
    agent_choice = 'AgentChoice'
    outguess_choice = 'OutguessChoice'

    agent_data = [agent_choice, outguess_choice]


class Game:
    def add_trial(self, agent_choice, reward):
        new_trial = pd.DataFrame(np.array([agent_choice, reward]).reshape(1, len(GameConstants.agent_data)),
                                 columns=GameConstants.agent_data)

        self.trials = pd.concat([self.trials, new_trial], ignore_index=True)
        self.trials[GameConstants.agent_choice] = self.trials[GameConstants.agent_choice].apply(np.int64)

        self.outguesser.update_model(
            self.trials[[GameConstants.agent_choice, GameConstants.outguess_choice]].values.astype(int))

        self.number_of_trials += 1

    def get_outguesser_response(self):
        return self.outguesser.predict_next_choice(
            self.trials[[GameConstants.agent_choice, GameConstants.outguess_choice]].values.astype(int))

    def get_agent_choices(self):
        return self.trials[GameConstants.agent_choice].values

    def get_outguesser_choices(self):
        return self.trials[GameConstants.outguess_choice].values

    def start(self):
        pass

    def stop(self):
        pass

    def __init__(self, outguesser):
        """
        :param outguesser: the algorithmic opponent that tries to detect biases and serial correlations in the choice of the agent
        """
        self.number_of_trials = 0
        self.outguesser = outguesser
        self.trials = pd.DataFrame(columns=GameConstants.agent_data)


UPDATE_ALGOS = {
    "choice_history": linear_choice_history_dependent_model,
    "choice_and_reward_history": choice_history_reward_history_model
}

PREDICTORS = {
    "choice_predictor": linear_choice_history_dependent_model_predictor,
    "choice_reward_predictor": choice_history_reward_history_model_predictor
}


def experiment(config_file):
    with open(config_file) as fp:
        config = json.load(fp)
    config.update({'config_file': config_file})
    print(config)
    config_experiment(**config)


def prepare_prior(update_model, history_length):
    if update_model == "choice_history":
        prior = np.zeros((history_length + 1,))
    elif update_model == "choice_and_reward_history":
        prior = np.zeros((2 * history_length + 1,))
    else:
        RuntimeError("Unknown update model {:s}".format(update_model))
    return prior


def config_experiment(game_type="no_feedback_v1", max_trials=10, history_dependence=1, record=False, data_dir='.',
                      config_file='', update_model='choice_history', predictor="choice_predictor"):
    def finish_game(**game_data):
        if record:
            agent_name = game_data["name"]
            device_type = game_data["device_type"]
            meta_data = GameMetaData(game_type)
            meta_data.add_config_file(config_file)
            meta_data.add_device_type(device_type)
            save_game(data_dir, meta_data, agent_name, game)

    prior = prepare_prior(update_model, history_dependence)
    update_algorithm = UPDATE_ALGOS[update_model]
    predictor = PREDICTORS[predictor]
    game = Game(
        Outguesser(update_algorithm, predictor, prior, record))
    game_variants[game_type](game, max_trials, finish_game)
