import ipywidgets as widgets
import numpy as np
from IPython.display import display


def feedback_v1(g, max_trials):
    def calculate_agent_score():
        return np.sum(g.get_agent_choices() != g.get_outguesser_choices())

    def on_button_clicked(b):
        outguesser_choice = g.get_outguesser_response()
        agent_choice = 2 * int(b.description) - 1
        g.add_trial(agent_choice, outguesser_choice)
        progress_bar.value += 1

        # out.close()

        with out:
            # print "Your choice: {}  Our choice: {}".format(agent_choice, outguesser_choice)
            pass
        out.clear_output(True)

        if g.number_of_trials == max_trials:
            # print("You did it :)")
            # print("Your score: {}   Shannons score: {}".format(calculate_agent_score(), g.number_of_trials-calculate_agent_score()))
            button0.close()
            button1.close()
            progress_bar.close()

    agent_name = widgets.Text(value='', placeholder='What\'s your name?', description='Name:', disabled=False)
    display(agent_name)

    button0 = widgets.Button(description='0')

    button1 = widgets.Button(description='1')

    progress_bar = widgets.IntProgress(value=0, min=0, max=max_trials, step=1, description='Progress',
                                       orientation='horizontal')
    out = widgets.Output(layout={'border': '1px solid black'})
    widget_container = widgets.Box([button0, button1, progress_bar, out])
    display(widget_container)

    button0.on_click(on_button_clicked)
    button1.on_click(on_button_clicked)


def calculate_agent_score(game):
    return np.sum(game.get_agent_choices() != game.get_outguesser_choices())


def calculate_outguesser_score(game):
    return game.number_of_trials - calculate_agent_score(game)


def get_score_widget(game):
    score_widget = widgets.Label(
        "Your final score: {}   My final score: {}".format(calculate_agent_score(game),
                                                           calculate_outguesser_score(game)))
    return score_widget


def get_thank_you_message():
    return widgets.Label("Thank you for participating :)")


def get_progress_bar(max_trials):
    return widgets.IntProgress(value=0, min=0, max=max_trials, step=1, description='Progress',
                               orientation='horizontal')


def get_buttons(on_button_clicked, descriptions=['0', '1']):
    button0 = widgets.Button(description=descriptions[0])
    button1 = widgets.Button(description=descriptions[1])
    button0.on_click(on_button_clicked)
    button1.on_click(on_button_clicked)
    buttons = widgets.HBox([button0, button1])
    return buttons


def no_feedback_v2(g, max_trials, finish_game):
    def get_agent_choice(button):
        outguesser_choice = g.get_outguesser_response()
        agent_choice = 2 * int(button.description) - 1
        g.add_trial(agent_choice, outguesser_choice)
        progress_bar.value += 1

        if g.number_of_trials == max_trials:
            game_area.close()
            display(get_thank_you_message())
            finish_game(**data_collector)

    def react_to_mobile_vs_desktop_info(button):
        data_collector.update({'device_type': button.description})
        mobile_vs_desktop_buttons.close()
        display(game_area)

    def react_to_name_entry(name_widget):
        data_collector.update({'name': name_widget.value})
        name_widget.close()
        display(mobile_vs_desktop_buttons)

    name_field = widgets.Text(value='', placeholder='Your name? <johnsmith>', description='Name:',
                              disabled=False)
    name_field.on_submit(react_to_name_entry)

    display(name_field)

    mobile_vs_desktop_buttons = get_buttons(react_to_mobile_vs_desktop_info, descriptions=['Mobile', 'Desktop'])

    choice_buttons = get_buttons(get_agent_choice)
    progress_bar = get_progress_bar(max_trials)
    game_area = widgets.VBox([choice_buttons, progress_bar])

    data_collector = {}


def feedback_v2(g, max_trials, finish_game):
    def get_agent_choice(button):
        outguesser_choice = g.get_outguesser_response()
        agent_choice = 2 * int(button.description) - 1
        g.add_trial(agent_choice, outguesser_choice)
        progress_bar.value += 1

        with prev_result:
            if agent_choice == outguesser_choice:
                print("You lose!")
            else:
                print("You win!")
            print("Your score: %d  My score: %d" % (calculate_agent_score(g), calculate_outguesser_score(g)))
        prev_result.clear_output(True)

        if g.number_of_trials == max_trials:
            game_area.close()
            display(get_thank_you_message())
            # display(get_final_score_message())
            finish_game(**data_collector)

    def react_to_mobile_vs_desktop_info(button):
        data_collector.update({'device_type': button.description})
        mobile_vs_desktop_buttons.close()
        display(game_area)

    def react_to_name_entry(name_widget):
        data_collector.update({'name': name_widget.value})
        name_widget.close()
        display(mobile_vs_desktop_buttons)

    name_field = widgets.Text(value='', placeholder='Your name? <johnsmith>', description='Name:',
                              disabled=False)
    name_field.on_submit(react_to_name_entry)

    display(name_field)

    mobile_vs_desktop_buttons = get_buttons(react_to_mobile_vs_desktop_info, descriptions=['Mobile', 'Desktop'])

    choice_buttons = get_buttons(get_agent_choice)
    progress_bar = get_progress_bar(max_trials)
    prev_result = widgets.Output(layout={'border': '1px solid black'})
    game_area = widgets.VBox([choice_buttons, progress_bar, prev_result])

    data_collector = {}


def no_feedback_v1(g, max_trials, finish_game):
    def get_agent_choice(button):
        outguesser_choice = g.get_outguesser_response()
        agent_choice = 2 * int(button.description) - 1
        g.add_trial(agent_choice, outguesser_choice)
        progress_bar.value += 1

        if g.number_of_trials == max_trials:
            game_area.close()
            display(get_thank_you_message())
            finish_game(**data_collector)

    def react_to_name_entry(name_widget):
        data_collector.update({'name': name_widget.value})
        name_widget.close()
        display(game_area)

    name_field = widgets.Text(value='', placeholder='Your name? <johnsmith>', description='Name:',
                              disabled=False)
    name_field.on_submit(react_to_name_entry)

    display(name_field)

    choice_buttons = get_buttons(get_agent_choice)
    progress_bar = get_progress_bar(max_trials)
    game_area = widgets.VBox([choice_buttons, progress_bar])

    data_collector = {}


game_variants = {
    "no_feedback_v1": no_feedback_v1,
    "no_feedback_v2": no_feedback_v2,
    "feedback_v1": feedback_v1,
    "feedback_v2": feedback_v2
}
