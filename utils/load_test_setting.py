from .settings import *
import os
import time
import argparse
from loguru import logger


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value


def parse_args():
    parser = argparse.ArgumentParser(description='Train watermarking network')

    parser.add_argument('--cfg_file',
                        help='experiment configure file name',
                        default="test_settings.json",
                        type=str)
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)

    args = parser.parse_args()

    return args


'''
params setting
'''
args = parse_args()
settings = JsonConfig()
settings.load_json_file(args.cfg_file)

with_diffusion = settings.with_diffusion

dataset_path = settings.dataset_path
batch_size = settings.batch_size
model_epoch = settings.model_epoch
strength_factor = settings.strength_factor
save_images_number = settings.save_images_number
lr = 1e-3
H, W, message_length = settings.H, settings.W, settings.message_length
noise_layers = settings.noise_layers
opt = settings.__json__

result_folder = "results/" + settings.result_folder