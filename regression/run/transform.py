'''
This file will be used to convert all checkpoint files from letes say 200 epochs to 1000 epochs so that
we don'tt have to repeat the experiments upto that point.
Take the epoch name from json files , and make respective checkpoints files with the epoch change name for
each of continouing experiments.
'''
import os
import sys
import argparse
from shutil import copyfile

sys.path.append(os.getcwd())
import time
import numpy as np
from src.utils.run_utils import get_list_pending_experiments
from src.utils.json_handling import get_sorted_dict, get_param_iterable
from src.utils.formatting import create_file_name, pretty_print_experiment

parser = argparse.ArgumentParser()

parser.add_argument('--json', '-j', type=str, nargs='+', help='Json Files', required=True)  # the json configuration
parser.add_argument('--epochs', '-e', type=int, required= True)

args = parser.parse_args()
json_files = args.json
pythoncommands = []
for json_file in json_files:
    print(json_file)
    d = get_sorted_dict(json_file)
    experiments = get_param_iterable(d)
    # print(experiments)
    for experiment in experiments:
    # we ahve problem and agent now , now comes the running loop
        cp_folder, cp_filename = create_file_name(experiment, sub_folder='checkpoints')
        output_file_name_cp = cp_folder + cp_filename + '.cpt'

        new_epochs = args.epochs
        if os.path.exists(output_file_name_cp):
            print(output_file_name_cp)
            new_experiment = experiment.copy()
            new_experiment['epochs'] = new_epochs
            f, fil = create_file_name(new_experiment, sub_folder = 'checkpoints')
            new_output_file = f + fil + '.cpt'
            print(new_output_file)
            copyfile(output_file_name_cp, new_output_file)




