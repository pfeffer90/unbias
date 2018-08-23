from game import Game, GameConstants


def test_initialize_game():
    unbias_dummy = None
    Game(unbias_dummy)


def test_add_trial_adds_a_row_with_the_last_trial_data():
    unbias_dummy = None
    agent_choice = 0
    reward = 1

    game = Game(unbias_dummy)
    game.add_trial(agent_choice, reward)

    last_trial_data = game.trials.tail(1)
    assert last_trial_data[GameConstants.agent_choice].values[0] == agent_choice
    assert last_trial_data[GameConstants.outguess_choice].values[0] == reward
