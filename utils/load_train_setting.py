from .settings import *
import os
import time
import argparse
from loguru import logger
from torch.utils.tensorboard import SummaryWriter


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
                        default="train_settings.json",
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
only_decoder = settings.only_decoder

project_name = settings.project_name
dataset_path = settings.dataset_path
epoch_number = settings.epoch_number
batch_size = settings.batch_size
save_images_number = settings.save_images_number
lr = settings.lr
H, W, message_length = settings.H, settings.W, settings.message_length,
noise_layers = settings.noise_layers
H_val, W_val, message_length_val = settings.H_val, settings.W_val, settings.message_length_val
encoder_weight = settings.encoder_weight
decoder_weight = settings.decoder_weight
opt = settings.__json__
'''
file preparing
'''
full_project_name = project_name + "_i{}_m{}".format(H, message_length)
# for noise in noise_layers:
#     full_project_name += "_" + noise.replace('[', '(').replace(']', ')')
is_resume = opt.get('resume_path', None) is not None
is_pretrain = opt.get('pretrain_path', None) is not None
assert ((not is_resume) or (not is_pretrain)) == True
if is_resume:
    result_folder = "results/" + opt['resume_path'] + "/"
else:
    result_folder = "results/" + time.strftime(full_project_name + "__%Y_%m_%d__%H_%M_%S", time.localtime()) + "/"
if not os.path.exists(result_folder): os.mkdir(result_folder)
if not os.path.exists(result_folder + "images/"): os.mkdir(result_folder + "images/")
if not os.path.exists(result_folder + "models/"): os.mkdir(result_folder + "models/")

# with open(result_folder + "/train_log.txt", "w") as file:
#     content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",
#                                                         time.localtime()) + "-----------------------\n"

#     for item in settings.get_items():
#         content += item[0] + " = " + str(item[1]) + "\n"

#     print(content)
#     file.write(content)
# with open(result_folder + "/val_log.txt", "w") as file:
#     content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",
#                                                         time.localtime()) + "-----------------------\n"
#     file.write(content)

logger.add(result_folder + "/train_log.txt")
content = '\n'
for item in settings.get_items():
    content += item[0] + " = " + str(item[1]) + "\n"
logger.info(content)

if is_resume:
    writer = SummaryWriter(result_folder + "tsbd_" + opt['resume_path'])
else:
    writer = SummaryWriter(result_folder + "tsbd_" + time.strftime(full_project_name + "__%Y_%m_%d__%H_%M_%S", time.localtime()))
