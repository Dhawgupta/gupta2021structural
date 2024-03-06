'''
Plotting Utilities for Python
'''
import numpy as np
import torch
import os, sys
import pickle as pkl
sys.path.append(os.getcwd())

from src.utils.formatting import create_file_name
from src.utils.json_handling import get_param_iterable_runs, get_param_iterable
import itertools


def pkl_loader(filename):
    with open(filename, 'rb') as fil:
        data = pkl.load(fil)
    return data

def pkl_saver(obj, filename):
    with open(filename, 'wb') as fil:
        pkl.dump(obj, fil)

def smoothen_runs(data, factor = 0.99):
    datatemp = data.reshape(-1)
    smooth_data = np.zeros_like(datatemp)
    smooth_data[0] = datatemp[0]
    for i in range(1, len(datatemp)):
        smooth_data[i] = smooth_data[i-1] * factor + (1-factor) * datatemp[i]
    return smooth_data

def fill_runs(exp_config, run):
    while len(run) != exp_config['epochs'] + 1:
        run.append(run[-1])
    return run



def load_different_runs(json_handle):
    '''
    Format for json handle : Should not have any lists except for seed list to load different runs
    '''
    training_accuracies = []
    test_accuracies = []
    valid_accuracies = []
    losses = []
    # get the list of params
    iterable = get_param_iterable(json_handle)
    for i in iterable:
        # print(i)
        folder, file = create_file_name(i)
        filename = folder + file + '.dw'
        # load the file
        try:
            arr = pkl_loader(filename)
            training_accuracies.append(fill_runs(i, arr['train-accuracy']))
            test_accuracies.append(fill_runs(i,arr['test-accuracy']))
            valid_accuracies.append(fill_runs(i,arr['valid-accuracy']))
            losses.append(np.array(arr['loss']))
        except:
            print('Run not valid')
            pass
    training_accuracies = np.array(training_accuracies)
    test_accuracies = np.array(test_accuracies)
    valid_accuracies = np.array(valid_accuracies)
    losses = np.array(losses)
    # losses = np.array(losses)

    return training_accuracies, test_accuracies, valid_accuracies, losses


def load_different_entropy_sparsity_runs(json_handle):
    '''
    only meant  to work for one seed, so need to convert entrop[y and sparsiryt intro proper arrays
    '''
    training_accuracies = []
    test_accuracies = []
    valid_accuracies = []
    entropies = []
    sparsity = []
    losses = []
    # get the list of params
    iterable = get_param_iterable(json_handle)
    for i in iterable:
        # print(i)
        folder, file = create_file_name(i)
        filename = folder + file + '.dw'
        # load the file
        try:
            arr = pkl_loader(filename)
            training_accuracies.append(fill_runs(i, arr['train-accuracy']))
            test_accuracies.append(fill_runs(i,arr['test-accuracy']))
            valid_accuracies.append(fill_runs(i,arr['valid-accuracy']))
            losses.append(np.array(arr['loss']))
            entropies.append(np.array(arr['entropy']['train']))
            sparsity.append(np.array(arr['sparsity']['train']))
        except:
            print('Run not valid')
            pass
    training_accuracies = np.array(training_accuracies)
    test_accuracies = np.array(test_accuracies)
    valid_accuracies = np.array(valid_accuracies)
    losses = np.array(losses)
    entropies = np.array(entropies)
    sparsity = np.array(sparsity)
    # losses = np.array(losses)

    return training_accuracies, test_accuracies, valid_accuracies, losses, entropies, sparsity

def load_different_runs_vf(json_handle):
    '''
    Format for json handle : Should not have any lists except for seed list to load different runs
    '''
    training_accuracies = []
    test_accuracies = []
    valid_accuracies = []
    losses = []
    vf_losses = []
    # get the list of params
    iterable = get_param_iterable(json_handle)
    for i in iterable:
        # print(i)
        folder, file = create_file_name(i)
        filename = folder + file + '.dw'
        # load the file
        try:
            arr = pkl_loader(filename)
            training_accuracies.append(fill_runs(i, arr['train-accuracy']))
            test_accuracies.append(fill_runs(i,arr['test-accuracy']))
            valid_accuracies.append(fill_runs(i,arr['valid-accuracy']))
            losses.append(np.array(arr['loss']))
            vf_losses.append(np.array(arr['vf_loss']))
        except:
            print('Run not valid')
            pass
    training_accuracies = np.array(training_accuracies)
    test_accuracies = np.array(test_accuracies)
    valid_accuracies = np.array(valid_accuracies)
    losses = np.array(losses)
    vf_losses = np.array(vf_losses)
    # losses = np.array(losses)

    return training_accuracies, test_accuracies, valid_accuracies, losses , vf_losses

# FIXME change data = 'valid' to use the valie data for finding best
def find_best(json_handle, data = 'valid',  key = None, metric = 'auc'):
    '''
    Format of json handle : Should have list for all the parameters we want to search across
    key : If Key is none, it will find the best across all parameters else it will find the best across the key poarameter
    metric : auc or end (end :means the last 10 %  performance)
    data : Find best wrt to the following parameter
    '''
    best_auc = - np.inf
    best_params = None
    best_run = {}
    best_data = {}
    iterable = get_param_iterable_runs(json_handle)
    # iterable of all params
    # print(iterable)
    for i in iterable:
        folder, file = create_file_name(i, 'processed')
        filename = folder + file + '.pcsd'

        if not os.path.exists(filename):
            print(i)
            raise Exception(f"Processed Data  File Not Found, please process data first for : {filename}")
        # load the data
        data_obj = pkl_loader(filename)
        train = data_obj['train']
        test = data_obj['test']
        valid = data_obj['valid']
        loss = data_obj['loss']
        vf_loss = None
        if 'vf_loss' in data_obj.keys():
            vf_loss = data_obj['vf_loss']
        # data = torch.load(filename)
        mean = data_obj[data]['mean']
        stderr = data_obj[data]['stderr']
        if metric == 'auc':
            auc = np.mean(mean)
        elif metric == 'last':
            lenMean = mean.shape[0]
            auc = np.mean(mean[:- lenMean // 10])
            # print(mean.shape)
            # print(mean[:-lenMean//10])
        else:
            print("Invalid metric, defaulting to AUC")
            auc = np.mean(mean)
        if auc > best_auc:
            best_auc = auc
            best_run['mean'] = mean
            best_run['stderr'] = stderr
            best_params = i
            best_data['train'] = train
            best_data['test'] = test
            best_data['valid'] = valid
            best_data['loss'] = loss

            if vf_loss is not None:
                best_data['vf_loss'] = vf_loss
    print(best_auc)
    return best_run, best_params, best_data

# FIXME fix this script for validatoin data
def find_best_key(json_handle,data = 'valid', key = None, metric = 'auc'):
    '''
    TODO incomplete for this repo
    Format of json handle : Should have list for all the parameters we want to search across
    key : If Key is none, it will find the best across all parameters else it will find the best across the key poarameter if key is list it will return teh best parameter for each configurations
    metric : auc or end (end :means the last 10 %  performance)
    '''
    best_auc = dict() #{} - np.inf
    best_params = dict() #{}None
    best_run = dict() #{}
    # get all the ke
    best_data = dict()
    if not isinstance(key, list):
        keys = json_handle[key]
        for k in keys:
            best_auc[k] = -np.inf
            best_params[k] = None
            best_run[k] = {}
            best_data[k] = dict()
        iterable = get_param_iterable_runs(json_handle)
        # iterable of all params
        # print(iterable)
        for i in iterable:
            folder, file = create_file_name(i, 'processed')
            filename = folder + file + '.pcsd'

            if not os.path.exists(filename):
                raise Exception(f"Processed Data  File Not Found, please process data first for : {filename}")
            # load the data
            data_obj = pkl_loader(filename)
            # data = torch.load(filename)

            mean = data_obj[data]['mean']
            stderr = data_obj[data]['stderr']
            auc = np.mean(mean)
            k = i[key]
            if auc > best_auc[k]:
                best_auc[k] = auc
                best_run[k]['mean'] = mean
                best_run[k]['stderr'] = stderr
                best_params[k] = i
                best_data[k] = data_obj

        return best_run, best_params, keys, best_data
    else:
        keys_list = []
        keys = []
        for k in key:
            keys_list.append(json_handle[k])
        iterators = itertools.product(*keys_list)
        for i in iterators:
            keys.append(i)
            best_auc[i] = -np.inf
            best_params[i] = None
            best_run[i] = dict()
            best_data[i] = dict()
        iterable = get_param_iterable_runs(json_handle)
        for i in iterable:
            folder, file = create_file_name(i, 'processed')
            filename = folder + file + '.pcsd'

            if not os.path.exists(filename):
                raise Exception(f"Processed Data  File Not Found, please process data first for : {filename}")
            # load the data
            data_obj = pkl_loader(filename)
            mean = data_obj[data]['mean']
            stderr = data_obj[data]['stderr']
            auc = np.mean(mean)
            val = []
            for k in key:
                val.append(i[k])
            index = tuple(val)
            if auc > best_auc[index]:
                best_auc[index] = auc
                best_run[index]['mean'] = mean
                best_run[index]['stderr'] = stderr
                best_params[index] = i
                best_data[index] = data_obj
        return best_run, best_params, keys, best_data

def get_key_from_dict(json_handle, key, key_key):
    list_ = json_handle[key]
    keys = []
    for l in list_:
        keys.append(l[key_key])
    return keys


# FIXME , fix this for valdation test
def find_best_key_key(json_handle,data = 'valid', key = None, key_key = None,  metric = 'auc'):
    '''
    TODO incomplete for this repo
    Format of json handle : Should have list for all the parameters we want to search across
    key : If Key is none, it will find the best across all parameters else it will find the best across the key poarameter if key is list it will return teh best parameter for each configurations
    metric : auc or end (end :means the last 10 %  performance)
    '''
    best_auc = dict() #{} - np.inf
    best_params = dict() #{}None
    best_run = dict() #{}
    # get all the ke
    best_data = dict()
    if not isinstance(key, list):
        keys = get_key_from_dict(json_handle, key, key_key)

        for k in keys:
            best_auc[k] = -np.inf
            best_params[k] = None
            best_run[k] = {}
            best_data[k] = dict()
        iterable = get_param_iterable_runs(json_handle)
        # iterable of all params
        # print(iterable)
        for i in iterable:
            folder, file = create_file_name(i, 'processed')
            filename = folder + file + '.pcsd'
            if not os.path.exists(filename):
                raise Exception(f"Processed Data  File Not Found, please process data first for : {filename}")
            # load the data
            data_obj = pkl_loader(filename)
            # data = torch.load(filename)

            mean = data_obj[data]['mean']
            stderr = data_obj[data]['stderr']
            auc = np.mean(mean)
            k = i[key][key_key]
            if auc > best_auc[k]:
                best_auc[k] = auc
                best_run[k]['mean'] = mean
                best_run[k]['stderr'] = stderr
                best_params[k] = i
                best_data[k] = data_obj

        return best_run, best_params, keys, best_data


def find_best_key_subkeys(json_handle, subkeys : list, data = 'valid', key = None, metric = 'auc'):
    '''
    TODO incomplete for this repo
    Format of json handle : Should have list for all the parameters we want to search across
    key : If Key is none, it will find the best across all parameters else it will find the best across the key poarameter if key is list it will return teh best parameter for each configurations
    metric : auc or end (end :means the last 10 %  performance)
    '''
    # make individual dictinary for all the subkeys

    best_auc = dict() #{} - np.inf
    best_params = dict() #{}None
    best_run = dict() #{}
    # get all the ke
    best_data = dict()
    if not isinstance(key, list):
        keys_list = []
        sub_keys = []
        for k in subkeys:
            keys_list.append(json_handle[k])
        iterators = itertools.product(*keys_list)
        keys = json_handle[key]
        for i in iterators:
            sub_keys.append(i)
            best_auc[i] = dict()
            best_params[i] = dict()
            best_run[i] = dict()
            best_data[i] = dict()
            for k in keys:
                best_auc[i][k] = -np.inf
                best_params[i][k] = None
                best_run[i][k] = {}
                best_data[i][k] = dict()
        iterable = get_param_iterable_runs(json_handle)

        for i in iterable:
            folder, file = create_file_name(i, 'processed')
            filename = folder + file + '.pcsd'

            if not os.path.exists(filename):
                raise Exception(f"Processed Data  File Not Found, please process data first for : {filename}")
            # load the data
            data_obj = pkl_loader(filename)
            # data = torch.load(filename)

            mean = data_obj[data]['mean']
            stderr = data_obj[data]['stderr']
            auc = np.mean(mean)
            val = []
            for k in subkeys:
                val.append(i[k])
            index = tuple(val)
            k = i[key]
            if auc > best_auc[index][k]:
                best_auc[index][k] = auc
                best_run[index][k]['mean'] = mean
                best_run[index][k]['stderr'] = stderr
                best_params[index][k] = i
                best_data[index][k] = data_obj

        return best_run, best_params, keys, sub_keys,  best_data
    else:
        raise NotImplementedError

def find_best_key_subkeys(json_handle, subkeys : list, data = 'valid', key = None, metric = 'auc'):
    '''
    TODO incomplete for this repo
    Format of json handle : Should have list for all the parameters we want to search across
    key : If Key is none, it will find the best across all parameters else it will find the best across the key poarameter if key is list it will return teh best parameter for each configurations
    metric : auc or end (end :means the last 10 %  performance)
    '''
    # make individual dictinary for all the subkeys

    best_auc = dict() #{} - np.inf
    best_params = dict() #{}None
    best_run = dict() #{}
    # get all the ke
    best_data = dict()
    if not isinstance(key, list):
        keys_list = []
        sub_keys = []
        for k in subkeys:
            keys_list.append(json_handle[k])
        iterators = itertools.product(*keys_list)
        keys = json_handle[key]
        for i in iterators:
            sub_keys.append(i)
            best_auc[i] = dict()
            best_params[i] = dict()
            best_run[i] = dict()
            best_data[i] = dict()
            for k in keys:
                best_auc[i][k] = -np.inf
                best_params[i][k] = None
                best_run[i][k] = {}
                best_data[i][k] = dict()
        iterable = get_param_iterable_runs(json_handle)

        for i in iterable:
            folder, file = create_file_name(i, 'processed')
            filename = folder + file + '.pcsd'

            if not os.path.exists(filename):
                raise Exception(f"Processed Data  File Not Found, please process data first for : {filename}")
            # load the data
            data_obj = pkl_loader(filename)
            # data = torch.load(filename)

            mean = data_obj[data]['mean']
            stderr = data_obj[data]['stderr']
            auc = np.mean(mean)
            val = []
            for k in subkeys:
                val.append(i[k])
            index = tuple(val)
            k = i[key]
            if auc > best_auc[index][k]:
                best_auc[index][k] = auc
                best_run[index][k]['mean'] = mean
                best_run[index][k]['stderr'] = stderr
                best_params[index][k] = i
                best_data[index][k] = data_obj

        return best_run, best_params, keys, sub_keys,  best_data
    else:
        raise NotImplementedError
