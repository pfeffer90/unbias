import time
from os import listdir
from time import strftime

import pandas as pd

DATE_FORMAT = "%Y%m%d"


def get_file_idx(filename, data_dir):
    file_list = filter(lambda s: s.startswith(filename), listdir(data_dir))
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
    prev_meta_data = pd.read_csv(meta_data_file_path)
    new_meta_data = pd.DataFrame.from_dict(meta_data_dict)
    updated_meta_data = pd.concat([prev_meta_data, new_meta_data])
    updated_meta_data.to_csv(meta_data_file_path, )


def save_to_file(data_dir, file_name, data_frame):
    data_frame.to_csv(data_dir + '/' + file_name)


class GameMetaData:
    def add_agent_name(self, agent_name):
        self.meta_data_dict.update({'AgentName': agent_name})

    def add_game_variant(self, game_variant):
        self.meta_data_dict.update({'GameVariant': game_variant})

    def add_game_idx(self, game_idx):
        self.meta_data_dict.update({'GameIdx': game_idx})

    def add_config_file(self, config_file):
        self.meta_data_dict.update({'ConfigFile': config_file})

    def add_choice_recording_location(self, choices_file):
        self.meta_data_dict.update({'ChoicesFile': choices_file})

    def add_inference_recording_location(self, model_paramters_file):
        self.meta_data_dict.update({'ModelFile': model_paramters_file})

    def get_meta_data_dict(self):
        return self.meta_data_dict

    def __init__(self):
        self.meta_data_dict = {'Date': time.strftime(DATE_FORMAT)}
