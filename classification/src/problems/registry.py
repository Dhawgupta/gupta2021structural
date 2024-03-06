from src.problems.mnist_flat import MnistFlat

def get_problem(prob_name):
    if prob_name == 'mnist_flat':
        return MnistFlat
    else:
        raise NotImplementedError