import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def get_random_rectangle_inside(image_shape, height_ratio, width_ratio, remain_h=None, remain_w=None):
    image_height = image_shape[2]
    image_width = image_shape[3]

    if remain_h is None:
        remaining_height = int(height_ratio * image_height)
    else:
        remaining_height = int(remain_h)
    if remain_w is None:
        remaining_width = int(width_ratio * image_width)
    else:
        remaining_width = int(remain_w)

    if remaining_height == image_height:
        height_start = 0
    else:
        height_start = np.random.randint(0, image_height - remaining_height)

    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width)

    return height_start, height_start + remaining_height, width_start, width_start + remaining_width


class Crop(nn.Module):

    def __init__(self, height_ratio, width_ratio):
        super(Crop, self).__init__()
        self.height_ratio = height_ratio
        self.width_ratio = width_ratio

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover

        h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
                                                                     self.width_ratio)
        mask = torch.zeros_like(image)
        mask[:, :, h_start: h_end, w_start: w_end] = 1

        return image * mask

class Cropout(nn.Module):

    def __init__(self, height_ratio, width_ratio):
        super(Cropout, self).__init__()
        self.height_ratio = height_ratio
        self.width_ratio = width_ratio

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover

        h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
                                                                     self.width_ratio)
        output = cover_image.clone()
        output[:, :, h_start: h_end, w_start: w_end] = image[:, :, h_start: h_end, w_start: w_end]
        return output

class Dropout(nn.Module):

    def __init__(self, prob):
        super(Dropout, self).__init__()
        self.prob = prob

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover

        rdn = torch.rand(image.shape).to(image.device)
        output = torch.where(rdn > self.prob * 1., cover_image, image)
        return output


class CropOnly(nn.Module):

    def __init__(self, height_ratio, width_ratio):
        super(CropOnly, self).__init__()
        self.height_ratio = height_ratio
        self.width_ratio = width_ratio

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover

        h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
                                                                     self.width_ratio)
        return image[:, :, h_start: h_end, w_start: w_end]


class CropResize(nn.Module):
    def __init__(self, lowerbound=0.4, upperbound=0.9):
        super(CropResize, self).__init__()
        self.ratio = [lowerbound, upperbound]

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        ratio = np.random.uniform(self.ratio[0], self.ratio[1])
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, ratio, ratio)
        crop_patch = image[:, :, h_start: h_end, w_start: w_end]
        return F.interpolate(crop_patch, size=(image.shape[2], image.shape[3]), mode='bicubic')


class CropIn(nn.Module):

    def __init__(self, height_ratio=0.4, width_ratio=0.8):
        super(CropIn, self).__init__()
        self.height_ratio = height_ratio
        self.width_ratio = width_ratio

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover

        h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
                                                                     self.width_ratio)
        output = image.clone()
        random_indices = torch.randperm(cover_image.size(0))
        cover_image_permuted = cover_image[random_indices]
        output[:, :, h_start: h_end, w_start: w_end] = cover_image_permuted[:, :, h_start: h_end, w_start: w_end]
        return output
