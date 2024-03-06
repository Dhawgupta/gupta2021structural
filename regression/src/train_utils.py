from src.utils.evaluations_utils import evaluate_performance
import torch
import numpy as np
from sklearn.model_selection import KFold
import copy, os, time, sys
from src.problems.registry import get_problem
from src.utils.formatting import create_file_name

def get_dataset(problem):
    X_train, y_train = problem.getTrainSet()
    X_valid, y_valid = problem.getValidSet()
    X_test, y_test = problem.getTestSet()

    X_train = problem.applyTransform(X_train)
    X_valid = problem.applyTransform(X_valid)
    X_test = problem.applyTransform(X_test)

    train = torch.utils.data.TensorDataset(X_train, y_train)
    valid = torch.utils.data.TensorDataset(X_valid, y_valid)
    test = torch.utils.data.TensorDataset(X_test, y_test)

    trainloader = problem.getBatchLoader(train)
    validloader = problem.getFullLoader(valid)
    testloader = problem.getFullLoader(test)
    trainloaderfull = problem.getFullLoader(train)

    return trainloader, validloader, testloader, trainloaderfull




def get_dataset_kfold(problem , kfold , foldno):
    X_train, y_train = problem.getTrainSet()
    X_test, y_test = problem.getTestSet()


    # x_train, y_train = trainset.train_data.type(torch.FloatTensor), trainset.train_labels
    X_train = problem.applyTransform(X_train)  # normalize and flatten

    # x_test, y_test = testset.test_data.type(torch.FloatTensor), testset.test_labels
    X_test = problem.applyTransform(X_test)

    # make the approprite train and validation split according to the fold number
    train_index, valid_index = list(kfold.split(X_train, y_train))[foldno]
    x_train_fold = X_train[train_index]
    x_valid_fold = X_train[valid_index]
    y_train_fold = y_train[train_index]
    y_valid_fold = y_train[valid_index]
    train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
    valid = torch.utils.data.TensorDataset(x_valid_fold, y_valid_fold)
    trainloader = problem.getBatchLoader(train)
    validloader = problem.getFullLoader(valid)  # need to use validation data as test data

    # make the test dataset
    test = torch.utils.data.TensorDataset(X_test, y_test)
    testloader = problem.getFullLoader(test)

    trainloader_full = problem.getFullLoader(train)

    return trainloader, validloader, testloader, trainloader_full

def evaluate_dataset_agent(agent, trainloader, validloader, testloader, training_accuracy, valid_accuracy, test_accuracy):
    eval_train = evaluate_performance(agent, trainloader)
    eval_valid = evaluate_performance(agent, validloader)
    eval_test = evaluate_performance(agent, testloader)
    # print(f"Losses: Train : {eval_train}, Valid : {eval_valid}, Test : {eval_test}")
    # store stats
    training_accuracy.append(eval_train)
    valid_accuracy.append(eval_valid)
    test_accuracy.append(eval_test)


class ExperimentUtils():
    def __init__(self, experiment):
        self.experiment = experiment
        self.seed = experiment['seed']
        self.set_seeds(self.seed)
        #self.pretrain = experiment["pretrain"]
        # self.folds = experiment['folds']
        # self.kfold = KFold(n_splits = self.folds)
        # self.foldno = experiment['foldno']
        # assert self.foldno < self.folds, "Fold Number can't be greater than the number of folds"
        self.params = copy.deepcopy(experiment)
        # Set feature Size and problem
        self.problem = get_problem(experiment["problem"])(self.params)
        self.params['in_features'] = self.problem.get_feature_size()
        self.device = torch.device("cpu")
        self.params["device"] = self.device


        self.folder, self.filename = create_file_name(self.experiment)
        if not os.path.exists(self.folder):
            time.sleep(2)
            try:
                os.makedirs(self.folder)
            except:
                pass
        self.output_file_name = self.folder + self.filename

        if os.path.exists(self.output_file_name + '.dw'):
            print("Run Already Complete - Ending Run")
            exit()

    def set_seeds(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1)

    def get_dataset(self):
        return get_dataset(self.problem)

def even_out_losses(backprop_loss, agent_loss):
    elements_per_layer = agent_loss[0].shape[0]
    if elements_per_layer > 1:
        for i in range(len(backprop_loss)):
            backprop_loss[i] = np.hstack([backprop_loss[i], np.zeros(elements_per_layer - 1)])

    for j in agent_loss:
        backprop_loss.append(j)

    return backprop_loss






