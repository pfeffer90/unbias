import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import display

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


def setup_game(g, max_trials):

    def calculate_agent_score():
        return np.sum(g.trials[GameConstants.agent_choice].values != g.trials[GameConstants.outguess_choice].values)


    def on_button_clicked(b):
        outguesser_choice = g.get_outguesser_response()
        agent_choice = int(b.description)
        g.add_trial(agent_choice, outguesser_choice)

        print "Your choice: {}  Our choice: {}".format(agent_choice, outguesser_choice)
        if g.number_of_trials == max_trials:
            print("You did it :)")
            print("Your score: {}   Shannons score: {}".format(calculate_agent_score(), g.number_of_trials-calculate_agent_score()))
            button0.close()
            button1.close()

    button0 = widgets.Button(description='0')

    button1 = widgets.Button(description='1')

    widget_container = widgets.Box([button0, button1])
    display(widget_container)

    button0.on_click(on_button_clicked)
    button1.on_click(on_button_clicked)

def genius(max_trials=10, history_dependence = 1):
    prior = np.zeros((history_dependence+1,))
    game = Game(Outguesser(simple_gradient_descent, maximum_a_posteriori, prior))
    setup_game(game, max_trials)

