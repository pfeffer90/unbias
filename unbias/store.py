from os import listdir
from time import strftime


def get_file_idx(filename, data_dir):
    file_list = filter(lambda s: s.startswith(filename), listdir(data_dir))
    if len(file_list) == 0:
        previous_max_index = 0
    else:
        previous_max_index = max(map(lambda s: int(s[-8:-4]), file_list))
    return previous_max_index + 1


def get_file_id(data_dir):
    date_string = strftime("%Y%m%d")
    idx = get_file_idx(date_string, data_dir)
    return date_string + "_" + "{0:04d}".format(idx) + ".csv"


def save_trials(data_dir, trials):
    file_id = get_file_id(data_dir)
    trials.to_csv(data_dir+'/'+file_id)
