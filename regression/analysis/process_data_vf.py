'''
This file will take in json files and process the data across different runs to store the summary
'''
import os, sys, time, copy
sys.path.append(os.getcwd())
import numpy as np

from src.utils.json_handling import get_sorted_dict , get_param_iterable_runs
from src.utils.formatting import create_file_name, create_folder
from analysis.utils import load_different_runs_vf, pkl_saver, pkl_loader

# read the arguments etc
if len(sys.argv) < 2:
    print("usage : python analysis/process_data.py <list of json files")
    exit()

json_files = sys.argv[1:] # all the json files

# convert all json files to dict
json_handles = [get_sorted_dict(j) for j in json_files]

def process_runs(runs):
    # get mean and std
    mean = np.mean(runs, axis = 0)
    stderr = np.std(runs , axis = 0) / np.sqrt(runs.shape[0])
    return mean , stderr



# currentl doesnt not handle frames
def process_data_interface(json_handles):
    for js in json_handles:
        runs = []
        iterables = get_param_iterable_runs(js)
        for i in iterables:
            folder, file = create_file_name(i, 'processed')
            create_folder(folder) # make the folder before saving the file
            filename = folder + file + '.pcsd'
            # check if file exists
            print(filename)
            if os.path.exists(filename):
                print("Processed")

            else:
                train, test, validation, loss, vf_loss = load_different_runs_vf(i)
                mean_train, stderr_train = process_runs(train)
                # train
                train_data = {
                    'mean' : mean_train,
                    'stderr' : stderr_train
                }
                
                # validation
                mean_valid, stderr_valid = process_runs(validation)
                valid_data = {
                    'mean': mean_valid,
                    'stderr': stderr_valid
                }

                # test
                mean_test, stderr_test = process_runs(test)
                test_data = {
                    'mean': mean_test,
                    'stderr': stderr_test
                }
                
                # loss
                mean_loss, stderr_loss = process_runs(loss)
                # mean_loss = []
                # stderr_loss = []
                loss_data = {
                    'mean': mean_loss,
                    'stderr': stderr_loss
                }

                # vf loss
                mean_loss, stderr_loss = process_runs(vf_loss)
                # mean_loss = []
                # stderr_loss = []
                vf_loss_data = {
                    'mean': mean_loss,
                    'stderr': stderr_loss
                }
                # save the things
                pkl_saver({
                        'train' : train_data,
                        'test' : test_data,
                        'valid' : valid_data, 
                        'loss' : loss_data,
                        'vf_loss': vf_loss_data
                    }, filename)


    # print(iterables)

if __name__ == '__main__':
    process_data_interface(json_handles)