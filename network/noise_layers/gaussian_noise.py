import numpy as np
import torch
import torch.nn as nn


class GN(nn.Module):

    def __init__(self, std, mean=0):
        super(GN, self).__init__()
        self.std = std
        self.mean = mean

    def gaussian_noise(self, image, mean, std):
        if std <= 15:
            std_used = std
        else:
            std_used = np.random.uniform(15, std)
        std_used /= 127.5
        noise = torch.Tensor(np.random.normal(mean, std_used, image.shape)).to(image.device)
        out = image + noise
        return out

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        return self.gaussian_noise(image, self.mean, self.std)
