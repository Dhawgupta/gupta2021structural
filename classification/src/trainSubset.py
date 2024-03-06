'''
python src/train.py jsonfile id
Removing pretraining from this file
'''
import os, sys, time, copy
sys.path.append(os.getcwd())
import numpy as np
import torch
import random
import logging
from src.problems.registry import get_problem
from src.agents.registry import get_agent
from src.utils.json_handling import get_sorted_dict, get_param_iterable
from src.utils.formatting import create_file_name
from src.utils.evaluations_utils import evaluate_performance
from analysis.utils import pkl_saver
import warnings
from src.trainutils import get_train_valid_test, evaluate_dataset_agent
warnings.filterwarnings("ignore", category=UserWarning)

# tell teh format of input
if len(sys.argv) < 3:
    print("usage : python src/main.py json_file idx")
    exit()
json_file = sys.argv[1]
idx = int(sys.argv[2])
# load the json file
d = get_sorted_dict(json_file)
experiments = get_param_iterable(d)
experiment = experiments[ idx % len(experiments)]

# set the seeds
seed = experiment['seed']
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.set_num_threads(1)

pretrain = experiment["pretrain"]
assert not pretrain , "train.py not meant for pretraining codes"
params = copy.deepcopy(experiment)
problem = get_problem(experiment["problem"])(params)

# set the features and actions
params['in_features'] =  problem.get_feature_size()
params['num_classes'] = problem.get_num_classes()

# set the device
device = torch.device(f"cpu")
params["device"] = device

agent = get_agent(experiment["node"], experiment["agent"])(params)

folder , filename = create_file_name(experiment)
if not os.path.exists(folder):
    time.sleep(2)
    try:
        os.makedirs(folder)
    except:
        pass


def get_output_filename(ex):
    folder, filename = create_file_name(ex)
    if not os.path.exists(folder):
        time.sleep(2)
        try:
            os.makedirs(folder)
        except:
            pass
    output_file_name = folder + filename
    return output_file_name

output_file_name = folder + filename
# Cut the run if already done
if os.path.exists(output_file_name + '.dw'):
    print("Run Already Complete - Ending Run")
    exit()

t_start = time.time()
updates = 0
logging.basicConfig(level=logging.INFO)
epochs = experiment['epochs']
training_accuracy = []
valid_accuracy = []
test_accuracy = []
losses = []
epoch_times = []
t_last_epoch = t_start
trainset = problem.getTrainSet()
testset = problem.getTestSet()
start_epoch = 0
trainloader , validloader, testloader, trainloaderFull = get_train_valid_test(problem)
# use the full batch for train evaluation
evaluate_dataset_agent(agent, trainloaderFull, validloader, testloader, training_accuracy, valid_accuracy, test_accuracy)
# Run Experiment
for e in range(start_epoch, epochs):
    losses_epoch = []
    change_subset = True
    for data in trainloader:
        x, y = data
        loss = agent.train(x.float(), y, update_subset=change_subset)
        change_subset = False
        losses_epoch.append(loss)
        if np.isnan(loss[0]):
            while len(training_accuracy) != epochs + 1:
                training_accuracy.append(training_accuracy[-1])
                valid_accuracy.append(valid_accuracy[-1])
                test_accuracy.append(test_accuracy[-1])
                losses.append(losses[-1])
            pkl_saver({
                'train-accuracy': training_accuracy,
                'valid-accuracy': valid_accuracy,
                'test-accuracy': test_accuracy,
                'loss': losses,
                'epoch-times': epoch_times,
                'nan': True
            }, output_file_name + '.dw')
            logging.info(f"Nan Encountered, Experiment Terminated {json_file} : {idx}, Time Taken : {time.time() - t_start}")
            exit()

    losses_epoch = np.array(losses_epoch)
    losses.append(np.mean(losses_epoch, axis = 0))
    evaluate_dataset_agent(agent, trainloaderFull, validloader, testloader, training_accuracy, valid_accuracy,test_accuracy)
    epoch_times.append(time.time() - t_last_epoch)
    t_last_epoch = time.time()
    if (e +1 ) % 100 == 0:
        # save
        print(f'{e+1}/{epochs} Done and Saving State')
        print(f"Train : {training_accuracy[-1]}, Valid : {valid_accuracy[-1]}, Test : {test_accuracy[-1]}")
        experiment_copy = copy.deepcopy(experiment)
        experiment_copy['epochs'] = e + 1
        new_output_filename = get_output_filename(experiment_copy)
        pkl_saver({
            'train-accuracy': training_accuracy,
            'valid-accuracy': valid_accuracy,
            'test-accuracy': test_accuracy,
            'loss': losses,
            'epoch-times': epoch_times,
            'epochs' : e+1,
            'nan': False
        }, new_output_filename + '.dw')

# print(training_accuracy)
pkl_saver({
    'train-accuracy': training_accuracy,
    'valid-accuracy': valid_accuracy,
    'test-accuracy': test_accuracy,
    'loss': losses,
    'epoch-times': epoch_times,
    'nan' : False
},output_file_name + '.dw')

logging.info(f"Experment Done {json_file} : {idx}, Time Taken : {time.time() - t_start}")

