from src.problems.mnist_flat import MnistFlat
from src.problems.boston_housing import BostonHousing
from src.problems.correlated import Correlated

def get_problem(prob_name):
    if prob_name == 'mnist_flat':
        return MnistFlat
    if prob_name == 'boston_housing':
        return BostonHousing
    if prob_name == 'correlated':
        return Correlated
    else:
        raise NotImplementedError