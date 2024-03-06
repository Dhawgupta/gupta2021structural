import torch
from src.utils.evaluations_utils import evaluate_performance, evaluate_performance_and_entropy, evaluate_performance_and_entropy_sparsity
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


def get_train_valid_test(problem):
    '''
    Get a split with train test and valid.
    '''
    trainset = problem.getTrainSet()
    testset = problem.getTestSet()

    x_train, y_train = trainset.train_data.type(torch.FloatTensor), trainset.train_labels
    x_train = problem.applyTransform(x_train)  # normalize and flatten

    x_test, y_test = testset.test_data.type(torch.FloatTensor), testset.test_labels
    x_test = problem.applyTransform(x_test)

    # make the approprite train and validation split according to the fold number
    splitper = 1/6
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=splitper, random_state=50, shuffle=True, stratify=y_train)

    train = torch.utils.data.TensorDataset(x_train, y_train)
    valid = torch.utils.data.TensorDataset(x_valid, y_valid)

    trainloader = problem.getTrainLoader(train)
    validloader = problem.getTestLoader(valid)  # need to use validation data as test data
    trainloaderFull = problem.getTestLoader(train)
    # make the test dataset
    test = torch.utils.data.TensorDataset(x_test, y_test)
    testloader = problem.getTestLoader(test)

    return trainloader, validloader, testloader, trainloaderFull

def get_dataset(problem , kfold , foldno):
    trainset = problem.getTrainSet()
    testset = problem.getTestSet()


    x_train, y_train = trainset.train_data.type(torch.FloatTensor), trainset.train_labels
    x_train = problem.applyTransform(x_train)  # normalize and flatten

    x_test, y_test = testset.test_data.type(torch.FloatTensor), testset.test_labels
    x_test = problem.applyTransform(x_test)

    # make the approprite train and validation split according to the fold number
    train_index, valid_index = list(kfold.split(x_train, y_train))[foldno]
    x_train_fold = x_train[train_index]
    x_valid_fold = x_train[valid_index]
    y_train_fold = y_train[train_index]
    y_valid_fold = y_train[valid_index]
    train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
    valid = torch.utils.data.TensorDataset(x_valid_fold, y_valid_fold)
    trainloader = problem.getTrainLoader(train)
    validloader = problem.getTestLoader(valid)  # need to use validation data as test data

    # make the test dataset
    test = torch.utils.data.TensorDataset(x_test, y_test)
    testloader = problem.getTestLoader(test)

    return trainloader, validloader, testloader

def evaluate_dataset_agent(agent, trainloader, validloader, testloader, training_accuracy, valid_accuracy, test_accuracy):
    eval_train = evaluate_performance(agent, trainloader)
    eval_valid = evaluate_performance(agent, validloader)
    eval_test = evaluate_performance(agent, testloader)

    # store stats
    training_accuracy.append(eval_train[0])
    valid_accuracy.append(eval_valid[0])
    test_accuracy.append(eval_test[0])

    print(f"Train : {training_accuracy[-1]}, Valid : {valid_accuracy[-1]}, Test : {test_accuracy[-1]}")

def evaluate_dataset_agent_entropy(agent, trainloader, validloader, testloader, training_accuracy, valid_accuracy, test_accuracy, entropies):
    # evalute the entopyt and accuracy of the coagents as well
    eval_train = evaluate_performance_and_entropy(agent, trainloader)
    eval_valid = evaluate_performance_and_entropy(agent, validloader)
    eval_test = evaluate_performance_and_entropy(agent, testloader)

    # store stats
    training_accuracy.append(eval_train[0])
    valid_accuracy.append(eval_valid[0])
    test_accuracy.append(eval_test[0])

    entropies['train'].append(eval_train[-1])
    entropies['valid'].append(eval_valid[-1])
    entropies['test'].append(eval_test[-1])

    print(f"Train : {training_accuracy[-1]}, Valid : {valid_accuracy[-1]}, Test : {test_accuracy[-1]}")

def evaluate_dataset_agent_entropy_sparsity(agent, trainloader, validloader, testloader, training_accuracy, valid_accuracy, test_accuracy, entropies, sparsities):
    # evalute the entopyt and accuracy of the coagents as well
    eval_train = evaluate_performance_and_entropy_sparsity(agent, trainloader)
    eval_valid = evaluate_performance_and_entropy_sparsity(agent, validloader)
    eval_test = evaluate_performance_and_entropy_sparsity(agent, testloader)

    # store stats
    training_accuracy.append(eval_train[0])
    valid_accuracy.append(eval_valid[0])
    test_accuracy.append(eval_test[0])

    entropies['train'].append(eval_train[-2])
    entropies['valid'].append(eval_valid[-2])
    entropies['test'].append(eval_test[-2])

    sparsities['train'].append(eval_train[-1])
    sparsities['valid'].append(eval_valid[-1])
    sparsities['test'].append(eval_test[-1])

    print(f"Train : {training_accuracy[-1]}, Valid : {valid_accuracy[-1]}, Test : {test_accuracy[-1]}")


class ValueFunction(nn.Module):
    def __init__(self, params):
        super(ValueFunction, self).__init__()
        self.in_features = params['in_features']
        self.model = nn.Sequential()
        self.model_specs = params['model_specification']
        self.num_layers = self.model_specs["num_layers"]
        self.activation = self.model_specs.get('activation', "relu")
        self.num_nodes_layer = None
        if self.num_layers > 0 :
            self.num_nodes_layer = params['units_layer']
        make_nn_layers(self.model, self.in_features, 1, self.num_layers, self.activation, self.num_nodes_layer)

    def forward(self, x):
        return self.model(x)


def get_activation( activation):
    if activation == 'relu':
        return nn.ReLU()
    if activation == 'sigmoid' :
        return nn.Sigmoid()
    if activation == 'tanh':
        return nn.Tanh()

def make_nn_layers(model, in_features, out, num_layers, activation , num_nodes_layer = None):

    if num_layers == 0:  # linear classifier
        model.add_module("0", nn.Linear(in_features[0], out_features=out))
    else:
        model.add_module("0", nn.Linear(in_features[0], num_nodes_layer))
        model.add_module("0-a",  get_activation(activation))
        for i in range(num_layers - 1):
            model.add_module(f"{i + 1}", nn.Linear(num_nodes_layer, num_nodes_layer))
            model.add_module((f"{i + 1}-a"), get_activation(activation))

        model.add_module(f"{num_layers}", nn.Linear(num_nodes_layer, out))
    return model