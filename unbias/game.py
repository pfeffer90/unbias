import numpy as np
import pandas as pd


class GameConstants:
    trial_idx = 'TrialIdx'
    agent_choice = 'AgentChoice'
    reward = 'Reward'

    agent_data = [trial_idx, agent_choice, reward]


class Game:
    def add_trial(self, agent_choice, reward):
        new_trial = pd.DataFrame(np.array([self.number_of_trials + 1, agent_choice, reward]).reshape(1, 3),
                                 columns=GameConstants.agent_data)

        self.trials = pd.concat([self.trials, new_trial])
        self.trials[GameConstants.agent_choice] = self.trials[GameConstants.agent_choice].apply(np.int64)
        self.number_of_trials += 1

    def provide_reward(self, agent_choice):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def __init__(self, unbiaser):
        """
        :param unbiaser: a map from trial history to reward that tries to remove bias and history dependence in the choice of the agent
        """
        self.number_of_trials = 0
        self.trials = pd.DataFrame(columns=GameConstants.agent_data)
