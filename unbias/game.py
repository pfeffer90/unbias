import numpy as np
import pandas as pd

from interfaces import game_variants
from outguesser import Outguesser, simple_gradient_descent, maximum_a_posteriori


class GameConstants:
    trial_idx = 'TrialIdx'
    agent_choice = 'AgentChoice'
    outguess_choice = 'OutguessChoice'

    agent_data = [trial_idx, agent_choice, outguess_choice]


class Game:
    def add_trial(self, agent_choice, reward):
        new_trial = pd.DataFrame(np.array([self.number_of_trials + 1, agent_choice, reward]).reshape(1, 3),
                                 columns=GameConstants.agent_data)

        self.trials = pd.concat([self.trials, new_trial])
        self.trials[GameConstants.agent_choice] = self.trials[GameConstants.agent_choice].apply(np.int64)

        self.outguesser.update_model(self.trials[GameConstants.agent_choice].values)

        self.number_of_trials += 1

    def get_outguesser_response(self):
        return self.outguesser.predict_next_choice(self.trials[GameConstants.agent_choice].values)

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


def genius(game_type="no_feedback_v1", max_trials=10, history_dependence=1):
    # get_user_info()
    prior = np.zeros((history_dependence + 1,))
    game = Game(Outguesser(simple_gradient_descent, maximum_a_posteriori, prior))
    game_variants[game_type](game, max_trials)
