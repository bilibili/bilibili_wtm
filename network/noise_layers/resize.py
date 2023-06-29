import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .crop import get_random_rectangle_inside


class Resize(nn.Module):
    def __init__(self, lowerbound=0.45, upperbound=0.8):
        super(Resize, self).__init__()
        self.ratio = [lowerbound, upperbound]

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        ratio = np.random.uniform(self.ratio[0], self.ratio[1])
        down_h = int(ratio * image.shape[2])
        down_w = int(ratio * image.shape[3])
        down_image = F.interpolate(image, size=(down_h, down_w), mode='bicubic')
        up_image = F.interpolate(down_image, size=(image.shape[2], image.shape[3]), mode='bicubic')
        return up_image


class ResizeCropout(nn.Module):
    def __init__(self, resize_ratio=0.5, back_ratio=0.67):
        super(ResizeCropout, self).__init__()
        assert resize_ratio <= back_ratio
        self.resize_ratio = resize_ratio
        self.back_ratio = back_ratio

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        down_h = int(self.resize_ratio * image.shape[2])
        down_w = int(self.resize_ratio * image.shape[3])
        back_h = int(self.back_ratio * image.shape[2])
        back_w = int(self.back_ratio * image.shape[3])
        down_image = F.interpolate(image, size=(down_h, down_w), mode='bicubic')
        background = F.interpolate(cover_image, size=(back_h, back_w), mode='bicubic')
        # permute background in batch
        random_indices = torch.randperm(background.size(0))
        background = background[random_indices]
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(background.shape,
                                                                     None, None, down_h, down_w)
        stitch = background.clone()
        stitch[:, :, h_start: h_end, w_start: w_end] = down_image
        up_image = F.interpolate(stitch, size=(image.shape[2], image.shape[3]), mode='bicubic')
        return up_image
