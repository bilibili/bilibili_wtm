# string input
# twice compression (same crf), same interval, (7,4) lcb
# logger, calculate precision + recall

import os
from loguru import logger
from datetime import datetime

from torch.utils.data import DataLoader
from utils import *
from network.Network import *

from utils.linear_block_code import *
from utils.save_images import save_per_frame
from utils.fix_seed import set_random_seed
from utils.video_attacks import video_compression_attack, video_compression_attack2

from sklearn.metrics import recall_score


def validation_twice_crf(network, save_path, valid_path='results/combine5_valid_continous'):
    # valid_cfg
    batch_size = 1
    strength_factor = 1
    message_length = 8040
    H = 1920
    W = 1072

    # codec args
    crf = str(28)
    resize_ratio = str(1.0)
    fps = 30
    compress_type = 'h264'

    # linear block coding args, (robust_length, information_length)
    information_length = 4
    robust_length = 7

    interval = 10
    total_length = 500

    # watermark
    s = 'bilibili@copyright'

    # IO args
    cover_folder = valid_path  # original Image folder
    encode_folder = os.path.join(save_path, 'encode')   # watermarked image folder
    compress_folder = os.path.join(save_path, 'compress')  # compressed image folder

    logger.info("target message is: " + s)

    set_random_seed(0)

    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = MBRSTestDataset(cover_folder, H, W)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    logger.info("Start Encoding...")

    test_result = {
        "psnr": 0.0,
        "ssim": 0.0
    }

    message_used = torch.tensor(stringToBitArray(s, 8)).to(device)
    message_unit_length = len(message_used) // information_length * robust_length  # start of tail clip in message
    message_tail_position = message_length // message_unit_length * message_unit_length  # repeat until the end
    repeat_time = message_length // message_unit_length
    G = np.array([[1, 1, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 1, 0, 0],
                [1, 0, 1, 0, 0, 1, 0],
                [0, 1, 1, 0, 0, 0, 1]])
    lbc = LinearBlockCode()
    lbc.setG(G)


    num = 0
    for i, images in enumerate(test_dataloader):
        image = images.to(device)
        message_root = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)
        message = torch.zeros_like(message_root)
        message[:, message_tail_position:] = message_root[:, message_tail_position:]
        message_used_np = message_used.view(1, -1, information_length).repeat(batch_size, 1, 1).cpu().numpy()
        robust_message_list = []
        for b_i in range(batch_size):
            robust_message_list.append(torch.Tensor(lbc.c(message_used_np[b_i])).view(1, -1))
        robust_message = torch.cat(robust_message_list, 0)
        robust_message = torch.tile(robust_message, (1, repeat_time))
        message[:, :message_tail_position] = robust_message

        '''
        test
        '''
        network.encoder_decoder.eval()
        network.discriminator.eval()

        with torch.no_grad():
            # use device to compute
            images, messages = images.to(network.device), message.to(network.device)
            if i % interval == 0:
                encoded_images = network.encoder_decoder.module.encoder(images, messages)
                encoded_images = images + (encoded_images - image) * strength_factor

            for j in range(batch_size):
                frame = i * batch_size + j
                if i % interval == 0:
                    # logger.info("save encoded frame {}", i)
                    save_per_frame(encoded_images[j], frame, encode_folder)
                else:
                    save_per_frame(images[j], frame, encode_folder)

            # psnr
            psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2).item()
            # ssim
            ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean").item()

        result = {
            "psnr": psnr,
            "ssim": ssim,
        }

        if i % interval == 0:
            for key in result:
                test_result[key] += float(result[key])

        num += 1
        if num == total_length:
            break

    print(num)
    '''
    test results
    '''
    content = "Average :"
    for key in test_result:
        content += key + "=" + str(test_result[key] / (num // interval)) + ","

    logger.info(content)
    return_psnr = - test_result["psnr"] / (num // interval)
    return_ssim = test_result["ssim"] / (num // interval)

    # compress
    intermediate_compress_folder = os.path.join(save_path, 'compress_intermediate')
    video_compression_attack(encode_folder, intermediate_compress_folder, crf, resize_ratio, fps, compress_type, logger, interval)
    video_compression_attack(intermediate_compress_folder, compress_folder, crf, resize_ratio, fps, compress_type, logger, interval)


    # decode
    set_random_seed(0)

    test_dataset = MBRSTestDataset(compress_folder, H, W)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    logger.info("Start Decoding :")

    num = 0
    result_list = []
    for i, compress_images in enumerate(test_dataloader):
        compress_images = compress_images.to(device)
        compress_images = F.interpolate(compress_images, size=(image.shape[2],image.shape[3]), mode='bicubic')
        # message = torch.Tensor(np.random.choice([0, 1], (compress_images.shape[0], message_length))).to(device)
        message_root = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)
        message = torch.zeros_like(message_root)
        message[:, message_tail_position:] = message_root[:, message_tail_position:]
        message_used_np = message_used.view(1, -1, information_length).repeat(batch_size, 1, 1).cpu().numpy()
        robust_message_list = []
        for b_i in range(batch_size):
            robust_message_list.append(torch.Tensor(lbc.c(message_used_np[b_i])).view(1, -1))
        robust_message = torch.cat(robust_message_list, 0)
        robust_message = torch.tile(robust_message, (1, repeat_time))
        message[:, :message_tail_position] = robust_message
        '''
        test
        '''
        network.encoder_decoder.eval()
        network.discriminator.eval()

        with torch.no_grad():
            # use device to compute
            compress_images, messages = compress_images.to(network.device), message.to(network.device)
            if i % interval == 0:
                decoded_messages = network.encoder_decoder.module.decoder(compress_images)
            else:
                continue

        useful_messages = decoded_messages[:, :message_tail_position]
        useful_messages = useful_messages.view(batch_size, repeat_time, message_unit_length)
        voted_messages = torch.mean(useful_messages, dim=1)
        # voted_messages = useful_messages[:, 0, :]
        '''
        decoded message error rate
        '''
        voted_messages = voted_messages.gt(0.5).int()
        
        # lbc error correction
        
        for b in range(batch_size):
            rs = voted_messages
            rs = rs.reshape(-1, lbc.n()).cpu().numpy()
            cs = np.zeros_like(rs)
            for j in range(len(rs)):
                cs[j] = lbc.syndromeDecode(rs[j])
            rm = cs[:, -lbc.k():].reshape(-1)
            res = bitArrayToString(rm)
            try:
                logger.info("message after decoding frame-{}: " + res, i * batch_size + b)
                result_list.append(res == s)
            except Exception as e:
                result_list.append(False)
                logger.info(e)

        num += 1

    y_true = [True for i in range(len(result_list))]

    recall = recall_score(y_true, result_list, average='binary')
    logger.info("recall: {}", recall)

    return return_psnr, return_ssim, recall


def validation_twice_crf_shift(network, save_path, valid_path='results/combine5_valid_continous'):
    # valid_cfg
    batch_size = 1
    strength_factor = 1
    message_length = 8040
    H = 1920
    W = 1072

    # codec args
    crf = str(27)
    resize_ratio = str(1.0)
    fps = 30
    compress_type = 'h264'

    # linear block coding args, (robust_length, information_length)
    information_length = 4
    robust_length = 7

    interval = 10
    total_length = 500

    # watermark
    s = 'bilibili@copyright'

    # IO args
    cover_folder = valid_path  # original Image folder
    encode_folder = os.path.join(save_path, 'encode')   # watermarked image folder
    compress_folder = os.path.join(save_path, 'compress')  # compressed image folder

    logger.info("target message is: " + s)

    set_random_seed(0)

    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = MBRSTestDataset(cover_folder, H, W)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    logger.info("Start Encoding...")

    test_result = {
        "psnr": 0.0,
        "ssim": 0.0
    }

    message_used = torch.tensor(stringToBitArray(s, 8)).to(device)
    message_unit_length = len(message_used) // information_length * robust_length  # start of tail clip in message
    message_tail_position = message_length // message_unit_length * message_unit_length  # repeat until the end
    repeat_time = message_length // message_unit_length
    G = np.array([[1, 1, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 1, 0, 0],
                [1, 0, 1, 0, 0, 1, 0],
                [0, 1, 1, 0, 0, 0, 1]])
    lbc = LinearBlockCode()
    lbc.setG(G)


    num = 0
    for i, images in enumerate(test_dataloader):
        image = images.to(device)
        message_root = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)
        message = torch.zeros_like(message_root)
        message[:, message_tail_position:] = message_root[:, message_tail_position:]
        message_used_np = message_used.view(1, -1, information_length).repeat(batch_size, 1, 1).cpu().numpy()
        robust_message_list = []
        for b_i in range(batch_size):
            robust_message_list.append(torch.Tensor(lbc.c(message_used_np[b_i])).view(1, -1))
        robust_message = torch.cat(robust_message_list, 0)
        robust_message = torch.tile(robust_message, (1, repeat_time))
        message[:, :message_tail_position] = robust_message

        '''
        test
        '''
        network.encoder_decoder.eval()
        network.discriminator.eval()

        with torch.no_grad():
            # use device to compute
            images, messages = images.to(network.device), message.to(network.device)
            if i % interval == 0:
                encoded_images = network.encoder_decoder.module.encoder(images, messages)
                encoded_images = images + (encoded_images - image) * strength_factor

            for j in range(batch_size):
                frame = i * batch_size + j
                if i % interval == 0:
                    # logger.info("save encoded frame {}", i)
                    save_per_frame(encoded_images[j], frame, encode_folder)
                else:
                    save_per_frame(images[j], frame, encode_folder)

            # psnr
            psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2).item()
            # ssim
            ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean").item()

        result = {
            "psnr": psnr,
            "ssim": ssim,
        }

        if i % interval == 0:
            for key in result:
                test_result[key] += float(result[key])

        num += 1
        if num == total_length:
            break

    print(num)
    '''
    test results
    '''
    content = "Average :"
    for key in test_result:
        content += key + "=" + str(test_result[key] / (num // interval)) + ","

    logger.info(content)
    return_psnr = - test_result["psnr"] / (num // interval)
    return_ssim = test_result["ssim"] / (num // interval)

    # compress
    intermediate_compress_folder = os.path.join(save_path, 'compress_intermediate')
    video_compression_attack(encode_folder, intermediate_compress_folder, crf, resize_ratio, fps, compress_type, logger, interval)
    video_compression_attack2(intermediate_compress_folder, compress_folder, crf, resize_ratio, fps, compress_type, logger, shift=interval // 2)


    # decode
    set_random_seed(0)

    test_dataset = MBRSTestDataset(compress_folder, H, W)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    logger.info("Start Decoding :")

    num = 0
    result_list = []
    for i, compress_images in enumerate(test_dataloader):
        compress_images = compress_images.to(device)
        compress_images = F.interpolate(compress_images, size=(image.shape[2],image.shape[3]), mode='bicubic')
        # message = torch.Tensor(np.random.choice([0, 1], (compress_images.shape[0], message_length))).to(device)
        message_root = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)
        message = torch.zeros_like(message_root)
        message[:, message_tail_position:] = message_root[:, message_tail_position:]
        message_used_np = message_used.view(1, -1, information_length).repeat(batch_size, 1, 1).cpu().numpy()
        robust_message_list = []
        for b_i in range(batch_size):
            robust_message_list.append(torch.Tensor(lbc.c(message_used_np[b_i])).view(1, -1))
        robust_message = torch.cat(robust_message_list, 0)
        robust_message = torch.tile(robust_message, (1, repeat_time))
        message[:, :message_tail_position] = robust_message
        '''
        test
        '''
        network.encoder_decoder.eval()
        network.discriminator.eval()

        with torch.no_grad():
            # use device to compute
            compress_images, messages = compress_images.to(network.device), message.to(network.device)
            if i % interval == (interval - interval // 2):
                decoded_messages = network.encoder_decoder.module.decoder(compress_images)
            else:
                continue

        useful_messages = decoded_messages[:, :message_tail_position]
        useful_messages = useful_messages.view(batch_size, repeat_time, message_unit_length)
        voted_messages = torch.mean(useful_messages, dim=1)
        # voted_messages = useful_messages[:, 0, :]
        '''
        decoded message error rate
        '''
        voted_messages = voted_messages.gt(0.5).int()
        
        # lbc error correction
        
        for b in range(batch_size):
            rs = voted_messages
            rs = rs.reshape(-1, lbc.n()).cpu().numpy()
            cs = np.zeros_like(rs)
            for j in range(len(rs)):
                cs[j] = lbc.syndromeDecode(rs[j])
            rm = cs[:, -lbc.k():].reshape(-1)
            res = bitArrayToString(rm)
            try:
                logger.info("message after decoding frame-{}: " + res, i * batch_size + b)
                result_list.append(res == s)
            except Exception as e:
                result_list.append(False)
                logger.info(e)

        num += 1

    y_true = [True for i in range(len(result_list))]

    recall = recall_score(y_true, result_list, average='binary')
    logger.info("recall: {}", recall)

    return return_psnr, return_ssim, recall


def validation_twice_crf_repeat(network, save_path, valid_path='results/combine5_valid_continous', opt={}):
    # valid_cfg
    batch_size = 1
    strength_factor = 1
    message_length = opt.get('crf_val_msg_length', 256)
    H = 1920
    W = 1072

    # codec args
    crf = str(opt.get('crf', 32))
    resize_ratio = str(1.0)
    fps = 30
    compress_type = 'h264'

    interval = 10
    total_length = 500

    # watermark
    s = opt.get('wtm', 'bilibili@copyright')
    shift_crf_valid = opt.get('shift_crf_valid', False)
    if shift_crf_valid:
        decode_frame = (interval - interval // 2)
    else:
        decode_frame = 0

    # IO args
    cover_folder = valid_path  # original Image folder
    encode_folder = os.path.join(save_path, 'encode')   # watermarked image folder
    compress_folder = os.path.join(save_path, 'compress')  # compressed image folder

    logger.info("target message is: " + s)

    set_random_seed(0)

    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = MBRSTestDataset(cover_folder, H, W)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    logger.info("Start Encoding...")

    test_result = {
        "psnr": 0.0,
        "ssim": 0.0
    }

    message_used = torch.tensor(stringToBitArray(s, 8)).to(device)
    message_unit_length = len(message_used)  # start of tail clip in message
    message_tail_position = message_length // message_unit_length * message_unit_length  # repeat until the end
    repeat_time = message_length // message_unit_length

    num = 0
    for i, images in enumerate(test_dataloader):
        image = images.to(device)
        message_root = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)
        message = torch.zeros_like(message_root)
        message[:, message_tail_position:] = message_root[:, message_tail_position:]
        robust_message = message_used.unsqueeze(0).expand(batch_size, -1)
        robust_message = torch.tile(robust_message, (1, repeat_time))
        message[:, :message_tail_position] = robust_message

        '''
        test
        '''
        network.encoder_decoder.eval()
        network.discriminator.eval()

        with torch.no_grad():
            # use device to compute
            images, messages = images.to(network.device), message.to(network.device)
            if i % interval == 0:
                encoded_images = network.encoder_decoder.module.encoder(images, messages)
                encoded_images = images + (encoded_images - image) * strength_factor

            for j in range(batch_size):
                frame = i * batch_size + j
                if i % interval == 0:
                    # logger.info("save encoded frame {}", i)
                    save_per_frame(encoded_images[j], frame, encode_folder)
                else:
                    save_per_frame(images[j], frame, encode_folder)

            # psnr
            psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2).item()
            # ssim
            ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean").item()

        result = {
            "psnr": psnr,
            "ssim": ssim,
        }

        if i % interval == 0:
            for key in result:
                test_result[key] += float(result[key])

        num += 1
        if num == total_length:
            break

    print(num)
    '''
    test results
    '''
    content = "Average :"
    for key in test_result:
        content += key + "=" + str(test_result[key] / (num // interval)) + ","

    logger.info(content)
    return_psnr = - test_result["psnr"] / (num // interval)
    return_ssim = test_result["ssim"] / (num // interval)

    # compress
    intermediate_compress_folder = os.path.join(save_path, 'compress_intermediate')
    video_compression_attack(encode_folder, intermediate_compress_folder, crf, resize_ratio, fps, compress_type, logger, interval)
    if shift_crf_valid:
        video_compression_attack2(intermediate_compress_folder, compress_folder, crf, resize_ratio, fps, compress_type, logger, shift=interval // 2)
    else:
        video_compression_attack(intermediate_compress_folder, compress_folder, crf, resize_ratio, fps, compress_type, logger, interval)


    # decode
    set_random_seed(0)

    test_dataset = MBRSTestDataset(compress_folder, H, W)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    logger.info("Start Decoding :")

    num = 0
    result_list = []
    for i, compress_images in enumerate(test_dataloader):
        compress_images = compress_images.to(device)
        compress_images = F.interpolate(compress_images, size=(image.shape[2],image.shape[3]), mode='bicubic')
        # message = torch.Tensor(np.random.choice([0, 1], (compress_images.shape[0], message_length))).to(device)
        message_root = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)
        message = torch.zeros_like(message_root)
        message[:, message_tail_position:] = message_root[:, message_tail_position:]
        robust_message = message_used.unsqueeze(0).expand(batch_size, -1)
        robust_message = torch.tile(robust_message, (1, repeat_time))
        message[:, :message_tail_position] = robust_message
        '''
        test
        '''
        network.encoder_decoder.eval()
        network.discriminator.eval()

        with torch.no_grad():
            # use device to compute
            compress_images, messages = compress_images.to(network.device), message.to(network.device)
            if i % interval == decode_frame:
                decoded_messages = network.encoder_decoder.module.decoder(compress_images)
            else:
                continue

        useful_messages = decoded_messages[:, :message_tail_position]
        useful_messages = useful_messages.view(batch_size, repeat_time, message_unit_length)
        voted_messages = torch.mean(useful_messages, dim=1)
        # voted_messages = useful_messages[:, 0, :]
        '''
        decoded message error rate
        '''
        voted_messages = voted_messages.gt(0.5).int()
        
        # lbc error correction
        
        for b in range(batch_size):
            rs = voted_messages[b].cpu().numpy()
            res = bitArrayToString(rs)
            try:
                logger.info("message after decoding frame-{}: " + res, i * batch_size + b)
                result_list.append(res == s)
            except Exception as e:
                result_list.append(False)
                logger.info(e)

        num += 1

    y_true = [True for i in range(len(result_list))]

    recall = recall_score(y_true, result_list, average='binary')
    logger.info("recall: {}", recall)

    return return_psnr, return_ssim, recall
