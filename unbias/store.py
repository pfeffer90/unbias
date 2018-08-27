import os
import time
from os import listdir
from time import strftime

import pandas as pd

DATE_FORMAT = "%Y%m%d"
AGENT_CHOICE_FILE_ID = "agent_choices_{:s}"
MODEL_PARAM_FILE_ID = "model_params_{:s}"
GAME_IDX_ENDING = "_{0:04d}.csv"


def get_file_idx(filename, data_dir):
    file_list = list(filter(lambda s: s.startswith(filename), listdir(data_dir)))
    if len(file_list) == 0:
        previous_max_index = 0
    else:
        previous_max_index = max(map(lambda s: int(s[-8:-4]), file_list))
    return previous_max_index + 1


def get_file_id(data_dir):
    date_string = strftime(DATE_FORMAT)
    idx = get_file_idx(date_string, data_dir)
    return date_string + "_" + "{0:04d}".format(idx) + ".csv"


def dump_meta_data(data_dir, meta_data_dict):
    meta_data_file = 'gamedata.csv'
    meta_data_file_path = data_dir + '/' + meta_data_file
    new_meta_data = pd.DataFrame.from_records([meta_data_dict])
    if not os.path.isfile(meta_data_file_path):
        updated_meta_data = new_meta_data
    else:
        prev_meta_data = pd.read_csv(meta_data_file_path, index_col=0)
        updated_meta_data = pd.concat([prev_meta_data, new_meta_data], ignore_index=True)
    updated_meta_data.to_csv(meta_data_file_path)


def save_to_file(data_dir, file_name, data_frame):
    data_frame.to_csv(data_dir + '/' + file_name)


def get_game_idx(data_dir, agent_name):
    agent_choices_part = AGENT_CHOICE_FILE_ID.format(agent_name)
    return get_file_idx(agent_choices_part, data_dir)


def get_agent_choice_file(data_dir, agent_name):
    agent_choices_part = AGENT_CHOICE_FILE_ID.format(agent_name)
    file_idx = get_file_idx(agent_choices_part, data_dir)
    return agent_choices_part + GAME_IDX_ENDING.format(file_idx)


def get_model_choice_file(data_dir, agent_name):
    model_param_part = MODEL_PARAM_FILE_ID.format(agent_name)
    file_idx = get_file_idx(model_param_part, data_dir)
    return model_param_part + GAME_IDX_ENDING.format(file_idx)


def save_game(data_dir, game_meta_data, agent_name, game):
    agent_choice_file = get_agent_choice_file(data_dir, agent_name)
    model_param_file = get_model_choice_file(data_dir, agent_name)
    game_idx = get_game_idx(data_dir, agent_name)
    save_to_file(data_dir, agent_choice_file, game.trials)
    save_to_file(data_dir, model_param_file, game.outguesser.recording_data_frame)

    game_meta_data.add_agent_name(agent_name)
    game_meta_data.add_game_idx(game_idx)
    game_meta_data.add_choice_recording_location(agent_choice_file)
    game_meta_data.add_inference_recording_location(model_param_file)
    game_meta_data.add_config_file(model_param_file)
    dump_meta_data(data_dir, game_meta_data.get_meta_data_dict())


class GameMetaData:
    def add_agent_name(self, agent_name):
        self.meta_data_dict.update({'AgentName': agent_name})

    def add_game_type(self, game_variant):
        self.meta_data_dict.update({'GameVariant': game_variant})

    def add_game_idx(self, game_idx):
        self.meta_data_dict.update({'GameIdx': game_idx})

    def add_config_file(self, config_file):
        self.meta_data_dict.update({'ConfigFile': config_file})

    def add_choice_recording_location(self, choices_file):
        self.meta_data_dict.update({'ChoicesFile': choices_file})

    def add_inference_recording_location(self, model_parameters_file):
        self.meta_data_dict.update({'ModelFile': model_parameters_file})

    def get_meta_data_dict(self):
        return self.meta_data_dict

    def __init__(self, game_type):
        self.meta_data_dict = {'Date': time.strftime(DATE_FORMAT)}
        self.add_game_type(game_type)
