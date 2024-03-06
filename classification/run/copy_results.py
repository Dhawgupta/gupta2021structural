"""
Deletes all experiments corresponding to the experiment files
Usage:
python run/delete_experiment.py -j <json_files.json>
"""

import argparse
import os
import sys

sys.path.append(os.getcwd())
from src.utils.json_handling import get_sorted_dict, get_param_iterable
from src.utils.formatting import create_file_name

parser = argparse.ArgumentParser()

parser.add_argument('--json', '-j', type = str ,nargs='+', help='Json Files', required=True) # the json configuration
parser.add_argument('--source','-s', type = str, help = 'the source directory')

args = parser.parse_args()
json_files = args.json
source_folder = args.source

json_handles = [get_sorted_dict(j) for j in json_files]

for json_file in json_files:
    d = get_sorted_dict(json_file)
    experiments = get_param_iterable(d)
    for exp in experiments:
        folder, filename = create_file_name(exp)
        # create experiment
        output_filename = folder + filename + '.dw'

        if os.path.exists(output_filename):
            os.remove(output_filename)
            print(f'{output_filename} Removed')

