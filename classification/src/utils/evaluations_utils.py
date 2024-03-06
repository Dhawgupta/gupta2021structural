import torch
import numpy as np

def evaluate_performance(agent, dataset,runs = 1 ):

    # evaluate performance on datase
    correct = 0
    total = 0
    # evaluate pefromance on the whole dataset in once.

    with torch.no_grad():
        for data in dataset:
            for r in range(runs):
                images, labels = data
                predicted = agent.get_predicition_class(images.float())
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    return correct / total, correct, total

def evaluate_performance_and_entropy(agent, dataset):
    # evaluate performance on datase
    correct = 0
    total = 0
    # evaluate pefromance on the whole dataset in once.
    entropy_each_layer = []
    with torch.no_grad():
        for data in dataset:
            images, labels = data
            predicted = agent.get_predicition_class(images.float())
            entropy = agent.network.all_softmax
            for p in entropy:
                p = p + 1e-9
                logp = torch.log(p)
                H_dataset = -(p * logp).sum(dim = 2)
                Hmean = H_dataset.mean(dim = 0)
                Hstd = H_dataset.std(dim = 0)
                entropy_each_layer.append(Hmean.numpy())
                if not (entropy_each_layer[-1] == entropy_each_layer[-1]).all():
                    print('Issues')
                    print("Lot of issues ")
                    print("encountered Nan")
                    exit()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total, correct, total, np.array(entropy_each_layer)

def evaluate_performance_and_entropy_sparsity(agent, dataset):
    correct = 0
    total = 0
    # evaluate pefromance on the whole dataset in once.
    entropy_each_layer = []
    sparsity_each_layer = []
    with torch.no_grad():
        for data in dataset:
            images, labels = data
            predicted = agent.get_predicition_class(images.float())
            softmax = agent.network.all_softmax
            for p in softmax:
                greedy_actions = p.argmax(dim = 2)
                sparsity = greedy_actions.float().abs().mean(dim = 1) # get the average as a number of nodes > 1
                p = p + 1e-9
                logp = torch.log(p)
                H_dataset = -(p * logp).sum(dim=2)
                Hmean = H_dataset.mean(dim=0)
                Hstd = H_dataset.std(dim=0)
                entropy_each_layer.append(Hmean.numpy())
                sparsity_each_layer.append(sparsity.mean().numpy())
                if not (entropy_each_layer[-1] == entropy_each_layer[-1]).all():
                    print('Issues')
                    print("Lot of issues ")
                    print("encountered Nan")
                    exit()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total, correct, total, np.array(entropy_each_layer), np.array(sparsity_each_layer)