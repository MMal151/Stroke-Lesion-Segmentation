import logging
import os
import random
import shutil

import yaml

CLASS_NAME = "[Util/Utils]"


def is_valid_dir(pth):
    return (pth is not None or pth != "") and os.path.exists(pth) and os.path.isdir(pth)


def is_valid_file(file_name):
    return (file_name is not None or file_name != "") and os.path.exists(file_name)


def is_valid_str(in_str):
    return in_str is not None and len(in_str) > 0 and in_str.strip() != ""


# Returns list of all possible files within the directory & its subdirectories'
def get_all_possible_files_paths(root: str, ext: str = ".png"):
    lgr = CLASS_NAME + "[get_all_possible_files_paths()]"

    assert is_valid_dir(root), f"{lgr}: Invalid root path. Either it doesn't exist or is not a directory."

    file_paths = set()
    for curr_dir, sub_dirs, files in os.walk(root):
        for f in files:
            if f.endswith(ext) and (not os.path.isdir(f)):
                file_paths.add(os.path.join(curr_dir, f))
    return sorted(file_paths)


# Returns all sub-dirs within a folder
# Input: root -> Path of the folder
#        mode -> full_path - Return physical path of the folder. Else just return the name of each subdirectory.
def get_all_possible_subdirs(root: str, mode=""):
    lgr = CLASS_NAME + "[get_all_possible_subdirs()]"

    if is_valid_dir(root):
        file_paths = set()
        for curr_dir, sub_dirs, _ in os.walk(root):
            for f in sub_dirs:
                if mode == "full_path":
                    file_paths.add(os.path.join(curr_dir, f))
                else:
                    file_paths.add(f)
        return sorted(file_paths)
    else:
        logging.warning(f"{lgr}: Invalid root path. Either it doesn't exist or is not a directory.")


def remove_dirs(path, incl=None):
    lgr = CLASS_NAME + "[remove_dirs()]"

    if type(path) == list:
        if incl is None:
            incl = ""
        files = []
        for i in path:
            if incl in i:
                logging.debug(f"{lgr}: Deleting directory {i}")
                shutil.rmtree(i, ignore_errors=True)
            else:
                files.append(i)
        return files
    else:
        logging.debug(f"{lgr}: Deleting directory {path}")
        shutil.rmtree(path)


# Convert string to tuple.
def str_to_tuple(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


# Generate list of filters using configured filter value.
def get_filters(min_filter, tot_filters):
    lgr = CLASS_NAME + "[get_filters()]"

    filters = [8, 16, 32, 64, 128]
    if tot_filters > 0:
        filters.clear()
        curr_filter = min_filter
        for i in range(0, tot_filters):
            filters.append(curr_filter)
            curr_filter *= 2
    else:
        logging.info(f"{lgr}: Invalid value of total number of filters. Returning default value [8, 16, 32, 64, 128]")
    return filters


# Generate random indexes in-between two ranges.
def get_random_index(min_idx, max_idx):
    i = random.randint(min_idx, max_idx)
    j = random.randint(min_idx, max_idx)

    while i == j:
        i, j = get_random_index(min_idx, max_idx)

    return i, j


def get_configurations(config_file="_config.yml"):
    with open(config_file, "r") as configFile:
        return yaml.safe_load(configFile)
