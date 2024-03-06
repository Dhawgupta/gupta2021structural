import torch

def evaluate_performance(agent, dataset,runs = 1 ):
    '''
    Changed to measure the regression loss defined by the agent
    '''
    with torch.no_grad():
        for data in dataset:
            for r in range(runs):
                X, y = data
                loss = agent.evaluate(X, y)

    return loss


