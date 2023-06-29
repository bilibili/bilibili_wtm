from . import *
from .Encoder_MP import Encoder_MP, Encoder_MP_Diffusion, Encoder_Rept, Encoder_Attn, Encoder_Hid, Encoder_AttnFc, Encoder_AttnFc_DnSp, Encoder_Haar
from .Decoder import Decoder, Decoder_Diffusion, Decoder_Rept, Decoder_Hid, Decoder_Attn, Decoder_AttnFC, Decoder_AttnFC_DnSp, Decoder_Diffusion_Avg, Decoder_Large, Decoder_Large2
from .Noise import Noise


class EncoderDecoder(nn.Module):
    '''
    A Sequential of Encoder_MP-Noise-Decoder
    '''

    def __init__(self, H, W, message_length, noise_layers, opt={}):
        super(EncoderDecoder, self).__init__()
        self.encoder = eval(opt.get('encoder_arch', 'Encoder_MP'))(H, W, message_length, opt=opt)
        self.noise = Noise(noise_layers)
        self.decoder = eval(opt.get('decoder_arch', 'Decoder'))(H, W, message_length, opt=opt)

    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        noised_image = self.noise([encoded_image, image])
        decoded_message = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message


class EncoderDecoder_Diffusion(nn.Module):
    '''
    A Sequential of Encoder_MP-Noise-Decoder
    '''

    def __init__(self, H, W, message_length, noise_layers, opt={}):
        super(EncoderDecoder_Diffusion, self).__init__()
        self.encoder = Encoder_MP_Diffusion(H, W, message_length, opt=opt)
        self.noise = Noise(noise_layers)
        decoder_avg = opt.get('decoder_avg', False)
        if decoder_avg:
            self.decoder = Decoder_Diffusion_Avg(H, W, message_length, opt=opt)
        else:
            self.decoder = Decoder_Diffusion(H, W, message_length, opt=opt)

    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        noised_image = self.noise([encoded_image, image])
        decoded_message = self.decoder(noised_image)

        return encoded_image, noised_image, decoded_message
