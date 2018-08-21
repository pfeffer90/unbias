import pandas as pd


class Game:
    def add_trial(self, agent_choice, reward):
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
        agent_data = ['TrialIdx', 'AgentChoice', 'Reward']
        self.number_of_trials = 0
        self.trials = pd.DataFrame(columns=agent_data)
