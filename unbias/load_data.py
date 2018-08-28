import pandas as pd
import os

METADATA_FILE_PREFIX = "game_metadata_"
METADATA_COLUMNS = ['AgentName', 'ChoicesFile', 'ConfigFile', 'Date', 'DeviceType', 'GameIdx', 'GameVariant', 'ModelFile']

def load_common_metadata(data_dir):
    file_list = list(filter(lambda s: s.startswith(METADATA_FILE_PREFIX), os.listdir(data_dir)))
    file_list = [data_dir + '/' + f for f in file_list]
    file_list.sort(key=os.path.getmtime)
    complete_metadata = pd.DataFrame(columns=METADATA_COLUMNS)
    for filename in file_list:
        with open (filename,'r') as f:
            complete_metadata = pd.concat([complete_metadata, pd.read_csv(f, usecols=METADATA_COLUMNS)], ignore_index=True)
    return complete_metadata
