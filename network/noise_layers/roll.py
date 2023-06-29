import torch
import torch.nn as nn
import numpy as np


class Roll(nn.Module):
    def __init__(self, shift_max=5):
        super(Roll, self).__init__()
        self.shift_max = shift_max

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        h_shift, w_shift = np.random.randint(low=-self.shift_max, high=self.shift_max, size=(2))

        shifted_image = torch.roll(image, shifts=(h_shift, w_shift), dims=(2, 3))
        return shifted_image


class RollTest(nn.Module):
    def __init__(self, shift=5):
        super(RollTest, self).__init__()
        self.shift = shift

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        h_shift, w_shift = np.random.choice((self.shift, -self.shift), size=(2))
        shifted_image = torch.roll(image, shifts=(h_shift, w_shift), dims=(2, 3))
        return shifted_image