from . import Identity
import torch.nn as nn
import numpy as np


class Combined(nn.Module):

    def __init__(self, list=None, p=None):
        super(Combined, self).__init__()
        if list is None:
            list = [Identity()]
        self.list = list
        self.p = p

    def forward(self, image_and_cover):
        if self.p is None:
            random_attack = np.random.choice(self.list)
        elif isinstance(self.p, list):
            random_attack = np.random.choice(self.list, p=(np.array(self.p) / np.array(self.p).sum()))
        return random_attack(image_and_cover), image_and_cover[1]
