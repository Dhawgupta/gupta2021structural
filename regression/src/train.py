'''
python src/train.py jsonfile id
'''
import os, sys, time
sys.path.append(os.getcwd())
import numpy as np
import logging
from src.agents.registry import get_agent
from src.train_utils import ExperimentUtils, evaluate_dataset_agent
from src.utils.json_handling import get_sorted_dict, get_param_iterable
from analysis.utils import pkl_saver
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning)
import statistics

if len(sys.argv) < 3:
    print("usage : python src/main.py json_file idx")
    exit()
json_file = sys.argv[1]
idx = int(sys.argv[2])
# Get experiment
d = get_sorted_dict(json_file)
experiments = get_param_iterable(d)
experiment = experiments[ idx % len(experiments)]

# get the experiment object
exp = ExperimentUtils(experiment)


# Cut the run if already done
if os.path.exists(exp.output_file_name + '.dw'):
    print("Run Already Complete - Ending Run")
    exit()


agent = get_agent(experiment["agent"])(exp.params)

t_start = time.time()
logging.basicConfig(level=logging.INFO)
epochs = experiment['epochs']
t_last_epoch = t_start
start_epoch = 0

# stats
train_losses = []
valid_losses = []
test_losses = []
losses = []
epoch_times = []

trainloader, validloader, testloader, trainloader_full = exp.get_dataset()

elapsed_training_steps = []
test_losses = []
train_losses = []
losses_buffer = []
trainloader_len = len(trainloader) / 200

#evaluate on full datasets for everything
for data in trainloader_full:
    X, y = data
    train_loss = agent.evaluate(X, y)
for data in testloader:
    X, y = data
    test_loss = agent.evaluate(X, y)
for data in validloader:
    X, y = data
    valid_loss = agent.evaluate(X, y)

test_losses.append(test_loss)
valid_losses.append(valid_loss)
train_losses.append(train_loss)

if experiment["problem"] == "correlated":
    for step,data in enumerate(trainloader):
        x, y = data
        loss = agent.train(x.float(), y)
        losses_buffer.append(loss[0]) # get the first loss, the acutal loss
        # Record a test loss and a mean train loss 200 times throughout training.
        if len(losses_buffer) >= trainloader_len :

            for data in testloader:
                X, y = data
                test_loss = agent.evaluate(X, y)
            for data in validloader:
                X, y = data
                valid_loss = agent.evaluate(X, y)

            test_losses.append(test_loss)
            valid_losses.append(valid_loss)
            train_losses.append(np.array(losses_buffer).mean())
            losses_buffer = []

    pkl_saver({
        'train': train_losses,
        'valid': valid_losses,
        'test': test_losses,
        'loss': valid_losses,
        'nan': False
    }, exp.output_file_name + '.dw')

else:
    for e in range(start_epoch, epochs):
        losses_epoch = []
        for data in trainloader_full:
            x, y = data
            loss = agent.train(x.float(), y)
            losses_epoch.append(loss)
            if np.isnan(loss[0]):
                while len(train_losses) != epochs + 1:
                    train_losses.append(train_losses[-1])
                    valid_losses.append(valid_losses[-1])
                    test_losses.append(test_losses[-1])
                    losses.append(losses[-1])
                pkl_saver({
                    'train': train_losses,
                    'valid': valid_losses,
                    'test': test_losses,
                    'loss': losses,
                    'epoch-times': epoch_times,
                    'nan': True
                }, exp.output_file_name + '.dw')
                logging.info(f"Nan Encountered, Experiment Terminated {json_file} : {idx}, Time Taken : {time.time() - t_start}")
                exit()

    losses_epoch = np.array(losses_epoch)
    losses.append(np.mean(losses_epoch, axis = 0))
    evaluate_dataset_agent(agent, trainloader_full, validloader, testloader, train_losses, valid_losses, test_losses)
    epoch_times.append(time.time() - t_last_epoch)
    t_last_epoch = time.time()

    pkl_saver({
        'train': train_losses,
        'valid': valid_losses,
        'test': test_losses,
        'loss': losses,
        'epoch-times': epoch_times,
        'nan': False
    }, exp.output_file_name + '.dw')

logging.info(f"Experment Done {json_file} : {idx}, Time Taken : {time.time() - t_start}")




