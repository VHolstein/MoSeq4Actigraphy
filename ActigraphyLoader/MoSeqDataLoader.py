import os
import glob
import scipy.io as sio
import numpy as np
from keypoint_moseq.initialize import *


# load data (add function for 1 big array)
class ActigraphyDataLoader:

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.mtl1_folder = os.path.join('actigraphy', 'processed', 'mtl1')

    def load_data(self, subjects: list = None, big_array: bool = False):
        print('done')
        # get all desired subjects in folder
        sub_paths = get_paths(self.base_dir, self.mtl1_folder, 'mtl1*.mat', subjects=subjects)
        # load files for each subject
        files = load_files(sub_paths=sub_paths)
        # return as dict if desired OR
        if big_array:
            x, mask, keys = merge_data(files)
            return x, mask, keys

        return files


def get_paths(folder, subfolder, extension, subjects=None):

    file_paths = {}

    if subjects:
        for subject in subjects:
            load_path = os.path.join(folder, subject, subfolder, extension)
            files = glob.glob(load_path)
            file_paths[subject] = files
    else:
        subjects = [f.name for f in os.scandir(folder) if f.is_dir()]
        for subject in subjects:
            load_path = os.path.join(folder, subject, subfolder, extension)
            files = glob.glob(load_path)
            file_paths[subject] = files

    return file_paths


def load_files(sub_paths: dict):

    pop_keys = []
    for key, item in sub_paths.items():
        if not item:
            pop_keys.append(key)
        else:
            fs = []
            for file in item:
                f = sio.loadmat(file)
                fs.append(f['dt2'])
            fs = np.concatenate(fs)
            sub_paths[key] = fs

    for to_pop in pop_keys:
        sub_paths.pop(to_pop)

    return sub_paths
