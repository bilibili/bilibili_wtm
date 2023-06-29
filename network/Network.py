from .Encoder_MP_Decoder import *
from .Discriminator import Discriminator, UNetDiscriminatorSN
import lpips
from loguru import logger


class Network:

    def __init__(self, H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion=False,
                 only_decoder=False, opt={}):
        # device
        self.device = device

        # network
        if not with_diffusion:
            self.encoder_decoder = EncoderDecoder(H, W, message_length, noise_layers, opt=opt).to(device)
        else:
            self.encoder_decoder = EncoderDecoder_Diffusion(H, W, message_length, noise_layers, opt=opt).to(device)

        self.discriminator = eval(opt.get('discriminator_arch', 'Discriminator'))().to(device)
        self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)
        self.discriminator = torch.nn.DataParallel(self.discriminator)

        # if only_decoder:
        #     for p in self.encoder_decoder.module.encoder.parameters():
        #         p.requires_grad = False

        # optimizer
        # print(lr)
        self.opt_encoder_decoder = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr)
        self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        # loss function
        self.cri_gan = GANLoss(opt.get('gan_loss_type', 'vanilla'), 1.0, 0.0, 1)
        self.which_enloss = opt.get('which_enloss', 'mse')
        if self.which_enloss == 'mse':
            fix_mse = opt.get('fix_mse', None)
            self.criterion_encode = TargetMSELoss(fix_mse)
        elif self.which_enloss == 'psnr':
            fix_value = opt.get('fix_value', 40)
            grad_weigthed = opt.get('grad_weigthed', None)
            psnr_mae = opt.get('psnr_mae', False)
            self.criterion_encode = PSNRLoss(2., fix_value, grad_weigthed, psnr_mae)
        elif self.which_enloss == 'l1':
            self.criterion_encode = nn.L1Loss().to(device)
        elif self.which_enloss == 'hinge_psnr':
            fix_value = opt.get('fix_value', 41)
            self.criterion_encode = HingePSNRLoss(2, fix_value)

        self.criterion_decode = TargetMSELoss(opt.get('target_decoder', None))

        # weight of encoder-decoder loss
        self.discriminator_weight = opt.get('discriminator_weight', None)
        self.encoder_weight = opt.get('encoder_weight', None)
        self.decoder_weight = opt.get('decoder_weight', None)
        self.lpips_weight = opt.get('lpips_weight', None)
        self.tv_weight = opt.get('tv_weight', None)
        if self.lpips_weight:
            self.criterion_lpips = lpips.LPIPS(net='vgg').to(device)
        if self.tv_weight:
            grad_weigthed = opt.get('grad_weigthed', None)
            self.criterion_tv = TVLoss(grad_weigthed)

    def train(self, images: torch.Tensor, messages: torch.Tensor):
        self.encoder_decoder.train()
        self.discriminator.train()

        with torch.enable_grad():
            # use device to compute
            images, messages = images.to(self.device), messages.to(self.device)
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

            '''
            train discriminator
            '''
            self.opt_discriminator.zero_grad()

            # real
            d_real = self.discriminator(images)
            d_real_loss = self.cri_gan(d_real, True, is_disc=True)
            d_real_loss.backward()

            # fake
            d_fake = self.discriminator(encoded_images.detach())
            d_fake_loss = self.cri_gan(d_fake, False, is_disc=True)
            d_fake_loss.backward()

            self.opt_discriminator.step()

            '''
            train encoder and decoder
            '''
            self.opt_encoder_decoder.zero_grad()

            # GAN : target label for encoded image should be "cover"(1)
            g_fake = self.discriminator(encoded_images)
            g_loss_on_discriminator = self.cri_gan(g_fake, True, is_disc=False)

            # RAW : the encoded image should be similar to cover image
            g_loss_on_encoder = self.criterion_encode(encoded_images, images)

            # RESULT : the decoded message should be similar to the raw message
            g_loss_on_decoder = self.criterion_decode(decoded_messages, messages)

            # full loss
            g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
                     self.decoder_weight * g_loss_on_decoder

            if self.lpips_weight:
                g_loss_lpips = self.criterion_lpips(encoded_images, images).mean()
                g_loss = g_loss + self.lpips_weight * g_loss_lpips

            if self.tv_weight:
                g_loss_tv = self.criterion_tv(encoded_images - images, images)
                g_loss = g_loss + self.tv_weight * g_loss_tv

            g_loss.backward()
            self.opt_encoder_decoder.step()

            # psnr
            psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

            # ssim
            ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")

        '''
        decoded message error rate
        '''
        error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

        result = {
            "error_rate": error_rate,
            "psnr": psnr,
            "ssim": ssim,
            "g_loss": g_loss,
            "g_loss_on_discriminator": g_loss_on_discriminator,
            "g_loss_on_encoder": g_loss_on_encoder,
            "g_loss_on_decoder": g_loss_on_decoder,
            "d_cover_loss": d_real_loss,
            "d_encoded_loss": d_fake_loss
        }
        if self.lpips_weight:
            result['g_loss_lpips'] = g_loss_lpips
        if self.tv_weight:
            result['g_loss_tv'] = g_loss_tv
        return result

    def train_only_decoder(self, images: torch.Tensor, messages: torch.Tensor):
        self.encoder_decoder.train()

        with torch.enable_grad():
            # use device to compute
            images, messages = images.to(self.device), messages.to(self.device)
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

            '''
            train encoder and decoder
            '''
            self.opt_encoder_decoder.zero_grad()

            # RESULT : the decoded message should be similar to the raw message
            g_loss_on_decoder = self.criterion_decode(decoded_messages, messages)

            g_loss = self.decoder_weight * g_loss_on_decoder

            g_loss.backward()
            self.opt_encoder_decoder.step()

            # psnr
            psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

            # ssim
            ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")

        '''
        decoded message error rate
        '''
        error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

        result = {
            "error_rate": error_rate,
            "psnr": psnr,
            "ssim": ssim,
            "g_loss": g_loss,
            "g_loss_on_discriminator": 0.,
            "g_loss_on_encoder": 0.,
            "g_loss_on_decoder": g_loss_on_decoder,
            "d_cover_loss": 0.,
            "d_encoded_loss": 0.
        }
        if self.lpips_weight:
            result['g_loss_lpips'] = 0
        if self.tv_weight:
            result['g_loss_tv'] = 0
        return result

    def validation(self, images: torch.Tensor, messages: torch.Tensor):
        self.encoder_decoder.eval()
        self.discriminator.eval()

        with torch.no_grad():
            # use device to compute
            images, messages = images.to(self.device), messages.to(self.device)
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

            '''
            validate discriminator
            '''
            # real
            d_real = self.discriminator(images)
            d_real_loss = self.cri_gan(d_real, True, is_disc=True)

            # fake
            d_fake = self.discriminator(encoded_images.detach())
            d_fake_loss = self.cri_gan(d_fake, False, is_disc=True)

            '''
            validate encoder and decoder
            '''

            # GAN : target label for encoded image should be "cover"(1)
            g_fake = self.discriminator(encoded_images)
            g_loss_on_discriminator = self.cri_gan(g_fake, True, is_disc=False)

            # RAW : the encoded image should be similar to cover image
            g_loss_on_encoder = self.criterion_encode(encoded_images, images)

            # RESULT : the decoded message should be similar to the raw message
            g_loss_on_decoder = self.criterion_decode(decoded_messages, messages)

            # full loss
            g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
                     self.decoder_weight * g_loss_on_decoder

            if self.lpips_weight:
                g_loss_lpips = self.criterion_lpips(encoded_images, images).mean()
                g_loss = g_loss + self.lpips_weight * g_loss_lpips

            if self.tv_weight:
                g_loss_tv = self.criterion_tv(encoded_images - images, images)
                g_loss = g_loss + self.tv_weight * g_loss_tv

            # psnr
            psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

            # ssim
            ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")

        '''
        decoded message error rate
        '''
        error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

        result = {
            "error_rate": error_rate,
            "psnr": psnr,
            "ssim": ssim,
            "g_loss": g_loss,
            "g_loss_on_discriminator": g_loss_on_discriminator,
            "g_loss_on_encoder": g_loss_on_encoder,
            "g_loss_on_decoder": g_loss_on_decoder,
            "d_cover_loss": d_real_loss,
            "d_encoded_loss": d_fake_loss
        }
        if self.lpips_weight:
            result['g_loss_lpips'] = g_loss_lpips
        if self.tv_weight:
            result['g_loss_tv'] = g_loss_tv

        return result, (images, encoded_images, noised_images, messages, decoded_messages)

    def decoded_message_error_rate(self, message, decoded_message):
        length = message.shape[0]

        message = message.gt(0.5)
        decoded_message = decoded_message.gt(0.5)
        error_rate = float(sum(message != decoded_message)) / length
        return error_rate

    def decoded_message_error_rate_batch(self, messages, decoded_messages):
        error_rate = 0.0
        batch_size = len(messages)
        for i in range(batch_size):
            error_rate += self.decoded_message_error_rate(messages[i], decoded_messages[i])
        error_rate /= batch_size
        return error_rate

    def save_model(self, path_encoder_decoder, path_discriminator, path_optimizor, epoch):
        torch.save(self.encoder_decoder.module.state_dict(), path_encoder_decoder)
        torch.save(self.discriminator.module.state_dict(), path_discriminator)
        torch.save({
            'epoch': epoch,
            'opt_encoder_decoder': self.opt_encoder_decoder.state_dict(),
            'opt_discriminator': self.opt_discriminator.state_dict(),
        }, path_optimizor)

    def resume_model(self, path_encoder_decoder, path_discriminator, path_optimizor):
        self.load_model(path_encoder_decoder, path_discriminator)
        optim_state_dict = torch.load(path_optimizor)
        self.opt_encoder_decoder.load_state_dict(optim_state_dict['opt_encoder_decoder'])
        self.opt_discriminator.load_state_dict(optim_state_dict['opt_discriminator'])
        return optim_state_dict['epoch']

    def load_model(self, path_encoder_decoder: str, path_discriminator: str, load_ed="ed"):
        if load_ed == "ed":
            self.load_model_ed(path_encoder_decoder)
        elif load_ed == "e":
            self.load_model_e(path_encoder_decoder)
        try:
            self.load_model_dis(path_discriminator)
        except Exception as e:
            logger.info(e)
            logger.info("init new discriminator instead")

    def load_model_ed(self, path_encoder_decoder: str):
        self.encoder_decoder.module.load_state_dict(torch.load(path_encoder_decoder))

    def load_model_e(self, path_encoder_decoder: str):
        ed_dict = torch.load(path_encoder_decoder)
        new_ed_dict = {k: v for k, v in ed_dict.items() if 'decoder' not in k}
        self.encoder_decoder.module.load_state_dict(new_ed_dict, strict=False)

    def load_model_dis(self, path_discriminator: str):
        self.discriminator.module.load_state_dict(torch.load(path_discriminator))


class PSNRLoss(nn.Module):

    def __init__(self, max_val=255, fix_value=40, masked=None, mae=False):
        super(PSNRLoss, self).__init__()
        self.scale = 10 / np.log(10)
        self.max_val = max_val
        self.fix_value = fix_value
        self.masked = masked
        if self.masked:
            self.get_grad = Get_gradient()
        self.mae = mae

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.masked:
            mask = 1. - (self.get_grad(target) / 3.) ** self.masked
            weight = mask.sum(dim=(1, 2, 3), keepdim=True)
            weight_mask = mask / weight
            mse = (((pred - target) ** 2) * weight_mask).sum(dim=(1, 2, 3))
        else:
            mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))
        psnr = - self.scale * torch.log(mse / (self.max_val**2) + 1e-8).mean()
        if self.fix_value:
            if self.mae:
                loss = (psnr - self.fix_value).abs()
            else:
                loss = (psnr - self.fix_value) ** 2
        else:
            loss = psnr
        return loss


class TargetMSELoss(nn.Module):
    def __init__(self, fix_value=None):
        super(TargetMSELoss, self).__init__()
        self.fix_value = fix_value
        self.MSE = nn.MSELoss()

    def forward(self, pred, target):

        mse = self.MSE(pred, target)
        if self.fix_value:
            loss = (mse - self.fix_value).abs()
        else:
            loss = mse
        return loss


class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x


class TVLoss(nn.Module):
    def __init__(self, masked=None):
        super(TVLoss, self).__init__()
        self.masked = masked
        if self.masked:
            self.get_grad = Get_gradient()

    def forward(self, pred, target=None):
        if self.masked is None:
            y_diff = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :]).mean()
            x_diff = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:]).mean()
        else:
            mask = 1. - (self.get_grad(target) / 3.) ** self.masked
            y_mask = mask[:, :, :-1, :]
            x_mask = mask[:, :, :, :-1]

            y_mask_sum = y_mask.sum(dim=(1, 2, 3), keepdim=True)
            y_weight = y_mask / y_mask_sum
            x_mask_sum = x_mask.sum(dim=(1, 2, 3), keepdim=True)
            x_weight = x_mask / x_mask_sum

            y_diff = (torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :]) * y_weight).sum(dim=(1, 2, 3)).mean()
            x_diff = (torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:]) * x_weight).sum(dim=(1, 2, 3)).mean()

        loss = y_diff + x_diff
        return loss


class HingePSNRLoss(nn.Module):

    def __init__(self, max_val=255, fix_value=41, half_range=1):
        super(HingePSNRLoss, self).__init__()
        self.scale = 10 / np.log(10)
        self.max_val = max_val
        self.fix_value = fix_value
        self.half_range = half_range
        self.relu = nn.ReLU()

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))
        psnr = - self.scale * torch.log(mse / (self.max_val**2) + 1e-8).mean()
        loss = (psnr - self.fix_value).abs() - self.half_range
        loss = self.relu(loss)
        return loss
