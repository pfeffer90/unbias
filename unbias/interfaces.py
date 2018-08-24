import ipywidgets as widgets
from IPython.display import display
import numpy as np

def feedback_v1(g, max_trials):

    def calculate_agent_score():
        return np.sum(g.get_agent_choices() != g.get_outguesser_choices())

    def on_button_clicked(b):
        outguesser_choice = g.get_outguesser_response()
        agent_choice = int(b.description)
        g.add_trial(agent_choice, outguesser_choice)
        progress_bar.value += 1

        #out.close()

        with out:
            print "Your choice: {}  Our choice: {}".format(agent_choice, outguesser_choice)

        out.clear_output(True)

        if g.number_of_trials == max_trials:
            print("You did it :)")
            print("Your score: {}   Shannons score: {}".format(calculate_agent_score(), g.number_of_trials-calculate_agent_score()))
            button0.close()
            button1.close()
            progress_bar.close()

    agent_name = widgets.Text(value='', placeholder='What\'s your name?', description='Name:', disabled=False)
    display(agent_name)

    
    button0 = widgets.Button(description='0')

    button1 = widgets.Button(description='1')

    progress_bar = widgets.IntProgress(value=0, min=0, max=max_trials, step=1, description='Progress', orientation='horizontal')
    out = widgets.Output(layout={'border': '1px solid black'})
    widget_container = widgets.Box([button0, button1, progress_bar, out])
    display(widget_container)

    button0.on_click(on_button_clicked)
    button1.on_click(on_button_clicked)



def no_feedback_v1(g, max_trials):

    def calculate_agent_score():
        return np.sum(g.get_agent_choices() != g.get_outguesser_choices())

    def on_button_clicked(b):
        outguesser_choice = g.get_outguesser_response()
        agent_choice = int(b.description)
        g.add_trial(agent_choice, outguesser_choice)
        progress_bar.value += 1

        out.clear_output(True)

        if g.number_of_trials == max_trials:
            print("You did it :)")
            print("Your score: {}   Shannons score: {}".format(calculate_agent_score(), g.number_of_trials-calculate_agent_score()))
            button0.close()
            button1.close()
            progress_bar.close()

    agent_name = widgets.Text(value='', placeholder='What\'s your name?', description='Name:', disabled=False)
    display(agent_name)

    
    button0 = widgets.Button(description='0')

    button1 = widgets.Button(description='1')

    progress_bar = widgets.IntProgress(value=0, min=0, max=max_trials, step=1, description='Progress', orientation='horizontal')
    out = widgets.Output(layout={'border': '1px solid black'})
    widget_container = widgets.Box([button0, button1, progress_bar, out])
    display(widget_container)

    button0.on_click(on_button_clicked)
    button1.on_click(on_button_clicked)


game_variants = {
    "no_feedback_v1" : no_feedback_v1
        }
