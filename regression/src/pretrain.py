'''
python src/train.py jsonfile id
'''
import os, sys, time
sys.path.append(os.getcwd())
import numpy as np
import logging
from src.optimizers.registry import  get_optimizer
from src.agents.registry import get_agent
from src.train_utils import ExperimentUtils, evaluate_dataset_agent, even_out_losses
from src.utils.json_handling import get_sorted_dict, get_param_iterable
from analysis.utils import pkl_saver
import warnings
from src.utils.backprop_to_coagents_utils import  copy_parameters
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# tell teh format of input
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

if exp.pretrain:
    backprop_agent = get_agent("backprop")(exp.params)
    backprop_epochs = experiment.get('backprop_epochs' , 5)

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

backprop_losses = []

if not exp.pretrain:
    logging.info(
        f"Only Run pretraining codes")
    exit()

if exp.pretrain:
    evaluate_dataset_agent(backprop_agent, trainloader_full, validloader, testloader, train_losses, valid_losses, test_losses)
    for e in range(backprop_epochs):
        losses_epoch = []
        for data in trainloader_full:
            x, y = data
            loss = backprop_agent.train(x.float(), y)
            agent.train_misc(loss[0]) # use only the global loss to update things
            losses_epoch.append(loss)
            if np.isnan(loss[0]):
                logging.info(
                    f"Nan Encountered, Experiment in Backprop Terminated {json_file} : {idx}, Time Taken : {time.time() - t_start}")
                exit()

        losses_epoch = np.array(losses_epoch)
        backprop_losses.append(np.mean(losses_epoch, axis=0))
        evaluate_dataset_agent(backprop_agent, trainloader_full, validloader, testloader, train_losses, valid_losses, test_losses)
        epoch_times.append(time.time() - t_last_epoch)
        t_last_epoch = time.time()

    # copy complete
    start_epoch = backprop_epochs
    # copy parameters and switch to a sgd version
    copy_parameters(backprop_agent, agent)
    # switch to SGD optimizer
    # reduce the learning rate by 1/2
    agent.optimizer = get_optimizer("sgd")(agent.network.parameters(), lr = agent.alpha / 128)
    # backprop_agent.optimizer = get_optimizer("sgd")(backprop_agent.network.parameters(), lr=agent.alpha / 128)

    print("redefined optimizer")

else:
    evaluate_dataset_agent(agent, trainloader_full, validloader, testloader, train_losses, valid_losses, test_losses)

for e in range(start_epoch, epochs):
    losses_epoch = []
    for data in trainloader_full:
        x, y = data
        loss = agent.train(x.float(), y)
        losses_epoch.append(loss)
        if np.isnan(loss[0]) or np.isinf(loss[0]):
            if exp.pretrain:
                losses = even_out_losses(backprop_losses, losses)
            while len(train_losses) != epochs + 1:
                # append inf
                train_losses.append(np.inf)
                valid_losses.append(np.inf)
                test_losses.append(np.inf)
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
    # process loss
    losses_epoch = np.array(losses_epoch)
    losses.append(np.mean(losses_epoch, axis = 0))
    # evaluate stats
    evaluate_dataset_agent(agent, trainloader_full, validloader, testloader, train_losses, valid_losses, test_losses)
    epoch_times.append(time.time() - t_last_epoch)
    t_last_epoch = time.time()

if exp.pretrain:
    losses = even_out_losses(backprop_losses, losses)

# plt.plot(train_losses, label = 'train')
# plt.plot(test_losses, label = 'test')
# plt.plot(valid_losses, label = 'valid')
# plt.legend()
# plt.ylim([0, 100])
# plt.show()

pkl_saver({
    'train': train_losses,
    'valid': valid_losses,
    'test': test_losses,
    'loss': losses,
    'epoch-times': epoch_times,
    'nan': False
}, exp.output_file_name + '.dw')

logging.info(f"Experment Done {json_file} : {idx}, Time Taken : {time.time() - t_start}")



