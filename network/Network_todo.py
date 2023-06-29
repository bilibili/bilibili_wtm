from .Encoder_MP_Decoder import *
from .Discriminator import Discriminator


class Network:

    def __init__(self, H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion=False,
                 only_decoder=False, opt={}):
        # device
        self.device = device

        # network
        if not with_diffusion:
            self.encoder_decoder = EncoderDecoder(H, W, message_length, noise_layers).to(device)
        else:
            self.encoder_decoder = EncoderDecoder_Diffusion(H, W, message_length, noise_layers).to(device)

        self.discriminator = Discriminator().to(device)

        self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)
        self.discriminator = torch.nn.DataParallel(self.discriminator)

        if only_decoder:
            for p in self.encoder_decoder.module.encoder.parameters():
                p.requires_grad = False

        # optimizer
        # print(lr)
        self.opt_encoder_decoder = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr)
        self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        # loss function
        self.cri_gan = GANLoss('vanilla', 1.0, 0.0, 1)
        self.criterion_MSE = nn.MSELoss().to(device)
        self.criterion_PSNR = PSNRLoss(2., 40)
        self.is_psnr = opt['is_psnr']

        # weight of encoder-decoder loss
        self.gan_weight = opt['gan_weight']
        self.encoder_weight = opt['encoder_weight']
        self.decoder_weight = opt['decoder_weight']

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
            gan_loss = self.cri_gan(g_fake, True, is_disc=False)

            # RAW : the encoded image should be similar to cover image
            if self.is_psnr:
                encoder_loss = self.criterion_PSNR(encoded_images, images)
            else:
                encoder_loss = self.criterion_MSE(encoded_images, images)

            # RESULT : the decoded message should be similar to the raw message
            decoder_loss = self.criterion_MSE(decoded_messages, messages)

            # full loss
            total_loss = self.gan_weight * gan_loss + self.encoder_weight * encoder_loss + self.decoder_weight * decoder_loss

            total_loss.backward()
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
            "total_loss": total_loss,
            "gan_loss": gan_loss,
            "encoder_loss": encoder_loss,
            "decoder_loss": decoder_loss,
            "d_real_loss": d_real_loss,
            "d_fake_loss": d_fake_loss
        }
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
            total_loss = self.criterion_MSE(decoded_messages, messages)

            total_loss.backward()
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
            "total_loss": total_loss,
            "gan_loss": 0.,
            "encoder_loss": 0.,
            "decoder_loss": 0.,
            "d_real_loss": 0.,
            "d_fake_loss": 0.
        }
        return result

    def validation(self, images: torch.Tensor, messages: torch.Tensor):
        self.encoder_decoder.eval()

        with torch.no_grad():
            # use device to compute
            images, messages = images.to(self.device), messages.to(self.device)
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

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
        }

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

    def load_model(self, path_encoder_decoder: str, path_discriminator: str):
        self.load_model_ed(path_encoder_decoder)
        self.load_model_dis(path_discriminator)

    def load_model_ed(self, path_encoder_decoder: str):
        self.encoder_decoder.module.load_state_dict(torch.load(path_encoder_decoder))

    def load_model_dis(self, path_discriminator: str):
        self.discriminator.module.load_state_dict(torch.load(path_discriminator))


class PSNRLoss(nn.Module):

    def __init__(self, max_val=255, fix_value=None):
        super(PSNRLoss, self).__init__()
        self.scale = 10 / np.log(10)
        self.max_val = max_val
        self.fix_value = fix_value

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))
        psnr = - self.scale * torch.log(mse / (self.max_val**2) + 1e-8).mean()
        if self.fix_value:
            loss = (psnr - self.fix_value) ** 2
        else:
            loss = psnr
        return loss
