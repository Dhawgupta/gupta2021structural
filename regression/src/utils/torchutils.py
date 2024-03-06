import torch
import numpy as np


def get_state(s, device = torch.device("cpu")):
    if len(s.shape) > 2 : # Minatar Env
        return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()
    else: # Normal Env's
        return torch.tensor(s, device = device).unsqueeze(0).float()
    

class TransformFlatten(object):
    def __call__(self, tensorT):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return tensorT.reshape(-1)
        

    def __repr__(self):
        return self.__class__.__name__ + '()'