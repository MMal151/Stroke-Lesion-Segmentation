import logging
import os
import random


def valid_dir(pth):
    return (pth is not None or pth != "") and os.path.exists(pth) and os.path.isdir(pth)


# Returns list of all possible files within the directory & its subdirectories'
def get_all_possible_files_paths(root: str, ext: str = ".png"):
    if valid_dir(root):
        file_paths = set()
        for curr_dir, sub_dirs, files in os.walk(root):
            for f in files:
                if f.endswith(ext) and (not os.path.isdir(f)):
                    file_paths.add(os.path.join(curr_dir, f))
        return sorted(file_paths)
    else:
        print("Invalid root path. Either it doesn't exist or is not a directory.")
        return -1


# Convert string to tuple.
def str_to_tuple(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


# Generate list of filters using configured filter value.
def get_filters(min_filter, tot_filters):
    filters = [8, 16, 32, 64, 128]
    if tot_filters > 0:
        filters.clear()
        curr_filter = min_filter
        for i in range(0, tot_filters):
            filters.append(curr_filter)
            curr_filter *= 2
    else:
        logging.info("Invalid value of total number of filters. Returning default value [8, 16, 32, 64, 128]")
    return filters


# Generate random indexes in-between two ranges.
def get_random_index(min, max):
    i = random.randint(min, max)
    j = random.randint(min, max)

    while i != j:
        i, j = get_random_index(min, max)

    return i, j
