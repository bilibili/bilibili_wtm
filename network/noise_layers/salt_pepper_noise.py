import torch
import torch.nn as nn
import numpy as np


class SP(nn.Module):

    def __init__(self, prob):
        super(SP, self).__init__()
        self.prob = prob

    def sp_noise(self, image, prob):
        b, c, h, w = image.shape
        signal_pct = prob
        noise_pct = (1 - prob)
        mask = torch.Tensor(np.random.choice((0, 1, 2), size=(b, 1, h, w), p=[signal_pct, noise_pct / 2., noise_pct / 2.])).to(image.device)
        mask = mask.repeat(1, c, 1, 1)
        #
        output = image.clone()
        output[mask == 1] = 1      # salt
        output[mask == 2] = -1     # pepper

        return output

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        return self.sp_noise(image, self.prob)
