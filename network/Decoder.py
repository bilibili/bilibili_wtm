from . import *


class Decoder(nn.Module):
    '''
    Decode the encoded image and get message
    '''

    def __init__(self, H, W, message_length, blocks=4, channels=64, opt={}):
        super(Decoder, self).__init__()

        channels = opt.get('decoder_ch', channels)
        blocks = opt.get('decoder_block', blocks)
        # stride_blocks = int(np.log2(H // int(np.sqrt(message_length))))
        stride_blocks = 4
        keep_blocks = max(blocks - stride_blocks, 0)

        self.first_layers = nn.Sequential(
            ConvBNRelu(3, channels),
            SENet_decoder(channels, channels, blocks=stride_blocks + 1),
            ConvBNRelu(channels * (2 ** stride_blocks), channels),
        )
        self.keep_layers = SENet(channels, channels, blocks=keep_blocks)

        self.final_layer = ConvBNRelu(channels, 1)

    def forward(self, noised_image):
        x = self.first_layers(noised_image)
        x = self.keep_layers(x)
        x = self.final_layer(x)
        x = x.view(x.shape[0], -1)
        return x


class Decoder_Large(nn.Module):
    '''
    Decode the encoded image and get message
    '''

    def __init__(self, H, W, message_length, blocks=4, channels=64, opt={}):
        super(Decoder_Large, self).__init__()

        channels = opt.get('decoder_ch', channels)
        blocks = opt.get('decoder_block', blocks)
        # stride_blocks = int(np.log2(H // int(np.sqrt(message_length))))
        stride_blocks = 4
        keep_blocks = max(blocks - stride_blocks, 0)

        # self.first_layers = nn.Sequential(
        #     ConvBNRelu(3, channels),
        #     SENet_decoder(channels, channels, blocks=stride_blocks + 1),
        #     ConvBNRelu(channels * (2 ** stride_blocks), channels),
        # )
        layer_list = [
            ConvBNRelu(3, channels),
            SENet_decoder(channels, channels, blocks=stride_blocks + 1)
        ]
        for s_b in range(stride_blocks):
            layer_list.append(ConvBNRelu(channels * (2 ** (stride_blocks - s_b)), channels * (2 ** (stride_blocks - s_b - 1))))
        self.first_layers = nn.Sequential(*layer_list)
        self.keep_layers = SENet(channels, channels, blocks=keep_blocks)

        self.final_layer = ConvBNRelu(channels, 1)

    def forward(self, noised_image):
        x = self.first_layers(noised_image)
        x = self.keep_layers(x)
        x = self.final_layer(x)
        x = x.view(x.shape[0], -1)
        return x


class Decoder_Large2(nn.Module):
    '''
    Decode the encoded image and get message
    '''

    def __init__(self, H, W, message_length, blocks=4, channels=64, opt={}):
        super(Decoder_Large2, self).__init__()

        channels = opt.get('decoder_ch', channels)
        blocks = opt.get('decoder_block', blocks)
        rpt_num = opt.get('decoder_rpt_num', 0)
        # stride_blocks = int(np.log2(H // int(np.sqrt(message_length))))
        stride_blocks = 4
        keep_blocks = max(blocks - stride_blocks, 0)

        # self.first_layers = nn.Sequential(
        #     ConvBNRelu(3, channels),
        #     SENet_decoder(channels, channels, blocks=stride_blocks + 1),
        #     ConvBNRelu(channels * (2 ** stride_blocks), channels),
        # )
        layer_list = [
            ConvBNRelu(3, channels),
            SENet_decoder(channels, channels, blocks=stride_blocks + 1)
        ]
        for s_b in range(stride_blocks):
            layer_list.append(ConvBNRelu(channels * (2 ** (stride_blocks - s_b)), channels * (2 ** (stride_blocks - s_b - 1))))
            for _ in range(rpt_num):
                layer_list.append(ConvBNRelu(channels * (2 ** (stride_blocks - s_b - 1)), channels * (2 ** (stride_blocks - s_b - 1))))
        self.first_layers = nn.Sequential(*layer_list)
        self.keep_layers = SENet(channels, channels, blocks=keep_blocks)

        self.final_layer = ConvBNRelu(channels, 1)

    def forward(self, noised_image):
        x = self.first_layers(noised_image)
        x = self.keep_layers(x)
        x = self.final_layer(x)
        x = x.view(x.shape[0], -1)
        return x


class Decoder_Diffusion(nn.Module):
    '''
    Decode the encoded image and get message
    '''

    def __init__(self, H, W, message_length, blocks=4, channels=64, diffusion_length=256, opt={}):
        super(Decoder_Diffusion, self).__init__()

        stride_blocks = 4
        # stride_blocks = int(np.log2(H // int(np.sqrt(diffusion_length))))

        self.diffusion_length = diffusion_length
        self.diffusion_size = int(self.diffusion_length ** 0.5)

        self.first_layers = nn.Sequential(
            ConvBNRelu(3, channels),
            SENet_decoder(channels, channels, blocks=stride_blocks + 1),
            ConvBNRelu(channels * (2 ** stride_blocks), channels),
        )
        self.keep_layers = SENet(channels, channels, blocks=1)

        self.final_layer = ConvBNRelu(channels, 1)

        self.message_layer = nn.Linear(self.diffusion_length, message_length)

    def forward(self, noised_image):
        x = self.first_layers(noised_image)
        x = self.keep_layers(x)
        x = self.final_layer(x)
        x = x.view(x.shape[0], -1)

        x = self.message_layer(x)
        return x


class Decoder_Diffusion_Avg(nn.Module):
    '''
    Decode the encoded image and get message
    '''

    def __init__(self, H, W, message_length, blocks=4, channels=64, diffusion_length=256, opt={}):
        super(Decoder_Diffusion_Avg, self).__init__()

        stride_blocks = 4
        # stride_blocks = int(np.log2(H // int(np.sqrt(diffusion_length))))

        self.diffusion_length = diffusion_length
        self.diffusion_size = int(self.diffusion_length ** 0.5)

        self.first_layers = nn.Sequential(
            ConvBNRelu(3, channels),
            SENet_decoder(channels, channels, blocks=stride_blocks + 1),
            ConvBNRelu(channels * (2 ** stride_blocks), channels),
        )
        self.keep_layers = SENet(channels, channels, blocks=1)

        self.final_layer = ConvBNRelu(channels, message_length)

        self.message_layer = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, noised_image):
        x = self.first_layers(noised_image)
        x = self.keep_layers(x)
        x = self.final_layer(x)

        x = self.message_layer(x).squeeze(-1).squeeze(-1)
        return x


class Decoder_Rept(nn.Module):
    '''
    Decode the encoded image and get message
    '''

    def __init__(self, H, W, message_length, blocks=4, channels=64, opt={}):
        super(Decoder_Rept, self).__init__()

        # stride_blocks = int(np.log2(H // int(np.sqrt(message_length))))
        stride_blocks = 4
        keep_blocks = max(blocks - stride_blocks, 0)

        self.first_layers = nn.Sequential(
            ConvBNRelu(3, channels),
            SENet(channels, channels, blocks=5),
            ConvBNRelu(channels, opt['fc_msg_num']),
        )
        self.keep_layers = SENet(opt['fc_msg_num'], opt['fc_msg_num'], blocks=keep_blocks)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.final_layer = nn.Linear(opt['fc_msg_num'], message_length)

    def forward(self, noised_image):
        x = self.first_layers(noised_image)
        x = self.keep_layers(x)
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.final_layer(x)
        return x


class Decoder_Attn(nn.Module):
    '''
    Insert a watermark into an image
    '''

    def __init__(self, H, W, message_length, blocks=4, channels=64, opt={}):
        super(Decoder_Attn, self).__init__()
        self.image_pre_layer = nn.Sequential(
            ConvBNRelu(3, channels),
            SENet(channels, channels, blocks=4),
            ConvBNRelu(channels, message_length),
        )

        is_softmax = opt.get('softmax', True)
        attention = [
            ConvBNRelu(3, message_length // 4),
            ConvBNRelu(message_length // 4, message_length),
            # SENet(message_length, message_length, blocks=1)
        ]
        if is_softmax:
            attention.append(nn.Softmax(dim=1))
        self.attention = nn.Sequential(*attention)

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, image):
        # first Conv part of Encoder
        image_feat = self.image_pre_layer(image)
        attn = self.attention(image)
        output = self.pool(image_feat * attn).squeeze(-1).squeeze(-1)
        return output


class Decoder_AttnFC(nn.Module):
    '''
    Insert a watermark into an image
    '''

    def __init__(self, H, W, message_length, blocks=4, channels=64, opt={}):
        super(Decoder_AttnFC, self).__init__()
        self.image_pre_layer = nn.Sequential(
            ConvBNRelu(3, channels),
            SENet(channels, channels, blocks=4),
            ConvBNRelu(channels, opt['fc_msg_num']),
        )

        is_softmax = opt.get('softmax', False)
        self.decoder_att_tail = opt.get('decoder_att_tail', False)
        attention = [
            ConvBNRelu(3, channels),
            ConvBNRelu(channels, opt['fc_msg_num']),
            # SENet(message_length, message_length, blocks=1)
        ]
        if is_softmax:
            attention.append(nn.Softmax(dim=1))
        self.attention = nn.Sequential(*attention)
        if self.decoder_att_tail:
            self.attention_tail = SENet(opt['fc_msg_num'], opt['fc_msg_num'], blocks=2)

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.msg_fc = nn.Sequential(
            nn.Linear(opt['fc_msg_num'], message_length)
        )

    def forward(self, image):
        # first Conv part of Encoder
        image_feat = self.image_pre_layer(image)
        attn = self.attention(image)
        attn = image_feat * attn
        if self.decoder_att_tail:
            attn = self.attention_tail(attn)
        output = self.pool(attn).squeeze(-1).squeeze(-1)
        output = self.msg_fc(output)
        return output


class Decoder_AttnFC_DnSp(nn.Module):
    '''
    Insert a watermark into an image
    '''

    def __init__(self, H, W, message_length, blocks=4, channels=64, opt={}):
        super(Decoder_AttnFC_DnSp, self).__init__()
        self.image_pre_layer = nn.Sequential(
            ConvBNRelu(3, channels),
            SENet_decoder(channels, channels, blocks=4 + 1),
            ConvBNRelu(channels * (2 ** 4), opt['fc_msg_num']),
            SENet(opt['fc_msg_num'], opt['fc_msg_num'], blocks=1),
        )

        is_softmax = opt.get('softmax', False)
        self.decoder_att_tail = opt.get('decoder_att_tail', False)
        self.is_attn_bias = opt.get('attn_bias', False)
        attention = [
            ConvBNRelu(3, channels // 8, 2),
            ConvBNRelu(channels // 8, channels // 4, 2),
            ConvBNRelu(channels // 4, channels // 2, 2),
            ConvBNRelu(channels // 2, opt['fc_msg_num'], 2),
            # SENet(message_length, message_length, blocks=1)
        ]
        if is_softmax:
            attention.append(nn.Softmax(dim=1))
        self.attention = nn.Sequential(*attention)
        if self.is_attn_bias:
            attention_bias = [
                ConvBNRelu(3, channels // 8, 2),
                ConvBNRelu(channels // 8, channels // 4, 2),
                ConvBNRelu(channels // 4, channels // 2, 2),
                ConvBNRelu(channels // 2, opt['fc_msg_num'], 2),
                # SENet(message_length, message_length, blocks=1)
            ]
            self.attention_bias = nn.Sequential(*attention_bias)
        if self.decoder_att_tail:
            self.attention_tail = SENet(opt['fc_msg_num'], opt['fc_msg_num'], blocks=2)

        pooling_type = opt.get('pooling_type', "average")
        if pooling_type == "average":
            self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        elif pooling_type == "max":
            self.pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.msg_fc = nn.Sequential(
            nn.Linear(opt['fc_msg_num'], message_length)
        )

    def forward(self, image):
        # first Conv part of Encoder
        image_feat = self.image_pre_layer(image)  # B 64 16 16
        attn = self.attention(image)  # B 64 16 16
        attn = image_feat * attn  # B 64 16 16
        if self.is_attn_bias:
            attn_bias = self.attention_bias(image)
            attn = attn + attn_bias
        if self.decoder_att_tail:
            attn = self.attention_tail(attn)
        output = self.pool(attn).squeeze(-1).squeeze(-1)  # B 64
        output = self.msg_fc(output)
        return output


class Decoder_Hid(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, H, W, message_length, opt={}):

        super(Decoder_Hid, self).__init__()
        self.channels = 64

        layers = [ConvBNRelu(3, self.channels)]
        for _ in range(7 - 1):
            layers.append(ConvBNRelu(self.channels, self.channels))

        # layers.append(block_builder(self.channels, config.message_length))
        layers.append(ConvBNRelu(self.channels, 256))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(256, 256)

    def forward(self, image_with_wm):
        x = self.layers(image_with_wm)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        x.squeeze_(3).squeeze_(2)
        x = self.linear(x)
        return x
