from torch.utils.data import DataLoader
from utils import *
from network.Network import *

from utils.load_train_setting import *
from utils.fix_seed import set_py_np_seed
from utils.utils import weight_strategy, adjust_learning_rate
from valid_crf_twice import validation_twice_crf, validation_twice_crf_shift, validation_twice_crf_repeat
'''
train
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = Network(H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion, only_decoder, opt)

train_dataset = MBRSDataset(os.path.join(dataset_path, "train"), H, W)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=opt['num_workers'], pin_memory=True)

val_dataset = MBRSValDataset(os.path.join(dataset_path, "validation"), H_val, W_val)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=opt['num_workers'], pin_memory=True)

if is_pretrain:
    EC_path = "results/" + opt['pretrain_path'] + "/models/EC_" + str(opt['pretrain_epoch']) + ".pth"
    D_path = "results/" + opt['pretrain_path'] + "/models/D_" + str(opt['pretrain_epoch']) + ".pth"
    load_ed = opt.get("load_ed", "ed")
    network.load_model(EC_path, D_path, load_ed)
    logger.info('Only load pretrained model of epoch {} since no optimizer states provided', opt['pretrain_epoch'])
if is_resume:
    EC_path = "results/" + opt['resume_path'] + "/models/EC_" + str(opt['resume_epoch']) + ".pth"
    D_path = "results/" + opt['resume_path'] + "/models/D_" + str(opt['resume_epoch']) + ".pth"
    path_optimizor = "results/" + opt['resume_path'] + "/models/ECD_optim.pth"
    load_epoch = network.resume_model(EC_path, D_path, path_optimizor)
    logger.info('Resume the state of epoch {}, where epoch {} is expected', load_epoch, opt['resume_epoch'])

logger.info("\nStart training : \n\n")
global_step = (len(train_dataloader) * (opt['resume_epoch'] + 1)) if is_resume else 0
start_epoch = (opt['resume_epoch'] + 1) if is_resume else 0
for epoch in range(start_epoch, epoch_number):

    running_result = {
        "error_rate": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "g_loss": 0.0,
        "g_loss_on_discriminator": 0.0,
        "g_loss_on_encoder": 0.0,
        "g_loss_on_decoder": 0.0,
        "g_loss_lpips": 0.0,
        "g_loss_tv": 0.0,
        "d_cover_loss": 0.0,
        "d_encoded_loss": 0.0
    }

    start_time = time.time()

    '''
    train
    '''
    num = 0
    decoder_epoch = opt.get('decoder_epoch', 0)
    increase_epoch = opt.get('increase_epoch', None)
    if epoch < decoder_epoch:
        adjust_learning_rate(network.opt_encoder_decoder, opt['lr_decoder_epoch'])
    else:
        adjust_learning_rate(network.opt_encoder_decoder, opt['lr'])

    if increase_epoch:
        if network.encoder_weight is not None:
            network.encoder_weight = weight_strategy(opt['encoder_weight'], epoch, decoder_epoch, increase_epoch, opt['w_strategy'])
        if network.discriminator_weight is not None:
            network.discriminator_weight = weight_strategy(opt['discriminator_weight'], epoch, decoder_epoch, increase_epoch, opt['w_strategy'])
        if network.lpips_weight is not None:
            network.lpips_weight = weight_strategy(opt['lpips_weight'], epoch, decoder_epoch, increase_epoch, opt['w_strategy'])
        if network.tv_weight is not None:
            network.tv_weight = weight_strategy(opt['tv_weight'], epoch, decoder_epoch, increase_epoch, opt['w_strategy'])
    for _, images, in enumerate(train_dataloader):
        image = images.to(device)
        message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)

        if not only_decoder:
            if epoch < decoder_epoch:
                result = network.train_only_decoder(image, message)
            else:
                result = network.train(image, message)
        else:
            result = network.train_only_decoder(image, message)

        for key in result:
            running_result[key] += float(result[key])
            writer.add_scalar('train_iter/' + key, result[key], global_step)

        num += 1
        global_step += 1
    writer.add_scalar('train_epoch/' + 'encoder_weight', network.encoder_weight, global_step)
    writer.add_scalar('train_epoch/' + 'encoder_decoder_lr', network.opt_encoder_decoder.param_groups[0]['lr'], global_step)
    writer.add_scalar('train_epoch/' + 'disciminator_lr', network.opt_discriminator.param_groups[0]['lr'], global_step)

    '''
    train results
    '''
    content = "Epoch " + str(epoch) + " - Time : " + str(int(time.time() - start_time)) + "s\n"
    recon_extract_score = -running_result['psnr'] / (running_result['error_rate'] + 1e-12)
    content += 'recon_extract_score' + "=" + str(recon_extract_score) + ","
    writer.add_scalar('train_epoch/' + 'recon_extract_score', recon_extract_score, global_step)
    for key in running_result:
        content += key + "=" + str(running_result[key] / num) + ","
        writer.add_scalar('train_epoch/' + key, running_result[key] / num, global_step)
    content += "\n"

    logger.info(content)
    '''
    validation
    '''
    if opt.get('valid_imagenet', True):
        set_py_np_seed(0)
        val_result = {
            "error_rate": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "g_loss": 0.0,
            "g_loss_on_discriminator": 0.0,
            "g_loss_on_encoder": 0.0,
            "g_loss_on_decoder": 0.0,
            "g_loss_lpips": 0.0,
            "g_loss_tv": 0.0,
            "d_cover_loss": 0.0,
            "d_encoded_loss": 0.0
        }

        start_time = time.time()

        saved_iterations = np.random.choice(np.arange(len(val_dataloader)), size=save_images_number, replace=False)
        saved_all = None

        num = 0
        for i, images in enumerate(val_dataloader):
            image = images.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length_val))).to(device)

            result, (images, encoded_images, noised_images, messages, decoded_messages) = network.validation(image, message)

            for key in result:
                val_result[key] += float(result[key])

            num += 1

            if i in saved_iterations:
                if saved_all is None:
                    saved_all = get_random_images(image, encoded_images, noised_images)
                else:
                    saved_all = concatenate_images(saved_all, image, encoded_images, noised_images)

        save_images(saved_all, epoch, result_folder + "images/", resize_to=(W, H))

        '''
        validation results
        '''
        content = "Epoch " + str(epoch) + " - Time : " + str(int(time.time() - start_time)) + "s\n"
        recon_extract_score = -val_result['psnr'] / (val_result['error_rate'] + 1e-12)
        content += 'recon_extract_score' + "=" + str(recon_extract_score) + ","
        writer.add_scalar('val_epoch/' + 'recon_extract_score', recon_extract_score, global_step)
        for key in val_result:
            content += key + "=" + str(val_result[key] / num) + ","
            writer.add_scalar('val_epoch/' + key, val_result[key] / num, global_step)
        content += "\n"

        logger.info(content)

    # crf twice
    if opt.get('valid_crf', False):
        shift_crf_valid = opt.get('shift_crf_valid', False)
        if shift_crf_valid:
            psnr, ssim, recall = validation_twice_crf_shift(network, result_folder, valid_path=opt['val_path'])
        else:
            psnr, ssim, recall = validation_twice_crf(network, result_folder, valid_path=opt['val_path'])
        # psnr, ssim, recall = validation_twice_crf_repeat(network, result_folder, valid_path=opt['val_path'], opt=opt)
        save_crf_img(epoch, opt['val_path'], result_folder, 'frame-0000.png')
        save_crf_img(epoch, opt['val_path'], result_folder, 'frame-0240.png')
        save_crf_img(epoch, opt['val_path'], result_folder, 'frame-0480.png')
        writer.add_scalar('crf_epoch/' + 'psnr', psnr, global_step)
        writer.add_scalar('crf_epoch/' + 'ssim', ssim, global_step)
        writer.add_scalar('crf_epoch/' + 'recall', recall, global_step)

    '''
    save model
    '''
    path_model = result_folder + "models/"
    path_encoder_decoder = path_model + "EC_" + str(epoch) + ".pth"
    path_discriminator = path_model + "D_" + str(epoch) + ".pth"
    path_optimizor = path_model + "ECD_optim.pth"
    network.save_model(path_encoder_decoder, path_discriminator, path_optimizor, epoch)
    set_py_np_seed(None)
