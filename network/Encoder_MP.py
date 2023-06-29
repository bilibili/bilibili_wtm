from . import *


class Encoder_MP(nn.Module):
    '''
    Insert a watermark into an image
    '''

    def __init__(self, H, W, message_length, blocks=4, channels=64, opt={}):
        super(Encoder_MP, self).__init__()
        self.H = H
        self.W = W
        channels = opt.get('encoder_ch', channels)
        blocks = opt.get('encoder_block', blocks)

        # message_convT_blocks = int(np.log2(H // int(np.sqrt(message_length))))
        message_convT_blocks = 4
        message_se_blocks = max(blocks - message_convT_blocks, 1)

        self.image_pre_layer = ConvBNRelu(3, channels)
        self.image_first_layer = SENet(channels, channels, blocks=blocks)

        self.message_pre_layer = nn.Sequential(
            ConvBNRelu(1, channels),
            ExpandNet(channels, channels, blocks=message_convT_blocks),
            SENet(channels, channels, blocks=message_se_blocks),
        )

        self.message_first_layer = SENet(channels, channels, blocks=blocks)

        self.after_concat_layer = ConvBNRelu(2 * channels, channels)

        self.final_layer = nn.Conv2d(channels + 3, 3, kernel_size=1)

    def forward(self, image, message):
        # first Conv part of Encoder
        image_pre = self.image_pre_layer(image)
        intermediate1 = self.image_first_layer(image_pre)

        # Message Processor
        # size = int(np.sqrt(message.shape[1]))
        # message_image = message.view(-1, 1, size, size)
        _, __, h, w = image.shape
        message_image = message.view(-1, 1, h//16, w//16)
        message_pre = self.message_pre_layer(message_image)
        intermediate2 = self.message_first_layer(message_pre)

        # concatenate
        concat1 = torch.cat([intermediate1, intermediate2], dim=1)

        # second Conv part of Encoder
        intermediate3 = self.after_concat_layer(concat1)

        # skip connection
        concat2 = torch.cat([intermediate3, image], dim=1)

        # last Conv part of Encoder
        output = self.final_layer(concat2)

        return output


class Encoder_MP_Diffusion(nn.Module):
    '''
    Insert a watermark into an image
    '''

    def __init__(self, H, W, message_length, blocks=4, channels=64, diffusion_length=256, opt={}):
        super(Encoder_MP_Diffusion, self).__init__()
        self.H = H
        self.W = W

        self.diffusion_length = diffusion_length
        self.diffusion_size = int(diffusion_length ** 0.5)

        self.image_pre_layer = ConvBNRelu(3, channels)
        self.image_first_layer = SENet(channels, channels, blocks=blocks)

        self.message_duplicate_layer = nn.Linear(message_length, self.diffusion_length)
        self.message_pre_layer_0 = ConvBNRelu(1, channels)
        self.message_pre_layer_1 = ExpandNet(channels, channels, blocks=4)
        self.message_pre_layer_2 = SENet(channels, channels, blocks=1)
        self.message_first_layer = SENet(channels, channels, blocks=blocks)

        self.after_concat_layer = ConvBNRelu(2 * channels, channels)

        self.final_layer = nn.Conv2d(channels + 3, 3, kernel_size=1)

    def forward(self, image, message):
        # first Conv part of Encoder
        image_pre = self.image_pre_layer(image)
        intermediate1 = self.image_first_layer(image_pre)

        # Message Processor (with diffusion)
        message_duplicate = self.message_duplicate_layer(message)
        message_image = message_duplicate.view(-1, 1, self.diffusion_size, self.diffusion_size)
        message_pre_0 = self.message_pre_layer_0(message_image)
        message_pre_1 = self.message_pre_layer_1(message_pre_0)
        message_pre_2 = self.message_pre_layer_2(message_pre_1)
        intermediate2 = self.message_first_layer(message_pre_2)

        # concatenate
        concat1 = torch.cat([intermediate1, intermediate2], dim=1)

        # second Conv part of Encoder
        intermediate3 = self.after_concat_layer(concat1)

        # skip connection
        concat2 = torch.cat([intermediate3, image], dim=1)

        # last Conv part of Network
        output = self.final_layer(concat2)

        return output


class Encoder_Attn(nn.Module):
    '''
    Insert a watermark into an image
    '''

    def __init__(self, H, W, message_length, blocks=4, channels=64, opt={}):
        super(Encoder_Attn, self).__init__()
        self.image_pre_layer = ConvBNRelu(3, channels)
        self.image_first_layer = SENet(channels, channels, blocks=blocks)

        is_softmax = opt.get('softmax', True)
        is_reduce = opt.get('reduce', True)
        self.is_reduce = is_reduce

        attention = [
            ConvBNRelu(3, message_length // 4),
            ConvBNRelu(message_length // 4, message_length),
            # SENet(message_length, message_length, blocks=1)
        ]
        if is_softmax:
            attention.append(nn.Softmax(dim=1))
        self.attention = nn.Sequential(*attention)

        self.after_concat_layer = ConvBNRelu((1 + channels) if self.is_reduce else (message_length + channels), channels)

        self.final_layer = nn.Conv2d(channels + 3, 3, kernel_size=1)

    def forward(self, image, message):
        # first Conv part of Encoder
        image_pre = self.image_pre_layer(image)
        intermediate1 = self.image_first_layer(image_pre)

        message = message * 2. - 1.
        attn = self.attention(image)
        intermediate2 = attn * message.unsqueeze(-1).unsqueeze(-1)
        if self.is_reduce:
            intermediate2 = torch.sum(intermediate2, dim=1, keepdim=True)

        # concatenate
        concat1 = torch.cat([intermediate1, intermediate2], dim=1)

        # second Conv part of Encoder
        intermediate3 = self.after_concat_layer(concat1)

        # skip connection
        concat2 = torch.cat([intermediate3, image], dim=1)

        # last Conv part of Encoder
        output = self.final_layer(concat2)

        return output


class Encoder_AttnFc(nn.Module):
    '''
    Insert a watermark into an image
    '''

    def __init__(self, H, W, message_length, blocks=4, channels=64, opt={}):
        super(Encoder_AttnFc, self).__init__()
        self.image_pre_layer = ConvBNRelu(3, channels)
        self.image_first_layer = SENet(channels, channels, blocks=blocks)

        is_softmax = opt.get('softmax', False)
        is_reduce = opt.get('reduce', False)
        self.encoder_att_tail = opt.get('encoder_att_tail', False)
        self.is_reduce = is_reduce

        attention = [
            ConvBNRelu(3, channels),
            ConvBNRelu(channels, opt['fc_msg_num']),
            # SENet(message_length, message_length, blocks=1)
        ]
        if is_softmax:
            attention.append(nn.Softmax(dim=1))
        self.attention = nn.Sequential(*attention)
        if self.encoder_att_tail:
            self.attention_tail = SENet(opt['fc_msg_num'], opt['fc_msg_num'], blocks=2)
        self.msg_fc = nn.Sequential(
            nn.Linear(message_length, opt['fc_msg_num'])
        )

        self.after_concat_layer = ConvBNRelu((1 + channels) if self.is_reduce else (opt['fc_msg_num'] + channels), channels)

        self.final_layer = nn.Conv2d(channels + 3, 3, kernel_size=1)

    def forward(self, image, message):
        # first Conv part of Encoder
        image_pre = self.image_pre_layer(image)
        intermediate1 = self.image_first_layer(image_pre)

        message = self.msg_fc(message * 2. - 1.)
        attn = self.attention(image)
        intermediate2 = attn * message.unsqueeze(-1).unsqueeze(-1)
        if self.encoder_att_tail:
            intermediate2 = self.attention_tail(intermediate2)
        if self.is_reduce:
            intermediate2 = torch.sum(intermediate2, dim=1, keepdim=True)

        # concatenate
        concat1 = torch.cat([intermediate1, intermediate2], dim=1)

        # second Conv part of Encoder
        intermediate3 = self.after_concat_layer(concat1)

        # skip connection
        concat2 = torch.cat([intermediate3, image], dim=1)

        # last Conv part of Encoder
        output = self.final_layer(concat2)

        return output


class Encoder_AttnFc_DnSp(nn.Module):
    '''
    Insert a watermark into an image
    '''

    def __init__(self, H, W, message_length, blocks=4, channels=64, opt={}):
        super(Encoder_AttnFc_DnSp, self).__init__()
        self.image_pre_layer = ConvBNRelu(3, channels)
        self.image_first_layer = SENet(channels, channels, blocks=blocks)

        is_softmax = opt.get('softmax', False)
        is_reduce = opt.get('reduce', False)
        self.encoder_att_tail = opt.get('encoder_att_tail', False)
        self.is_attn_bias = opt.get('attn_bias', False)
        self.is_fc_relu = opt.get('fc_relu', False)
        self.is_msg_norm = opt.get('msg_norm', True)
        self.is_reduce = is_reduce

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
        if self.encoder_att_tail:
            self.attention_tail = SENet(opt['fc_msg_num'], opt['fc_msg_num'], blocks=2)
        if self.is_fc_relu:
            self.msg_fc = nn.Sequential(
                nn.Linear(message_length, opt['fc_msg_num']),
                nn.ReLU(inplace=True)
            )
        else:
            self.msg_fc = nn.Sequential(
                nn.Linear(message_length, opt['fc_msg_num'])
            )

        self.message_pre_layer = nn.Sequential(
            ConvBNRelu(opt['fc_msg_num'], channels),
            ExpandNet(channels, channels, blocks=4),
            SENet(channels, channels, blocks=1),
            SENet(channels, channels, blocks=blocks)
        )

        self.after_concat_layer = ConvBNRelu(2 * channels, channels)

        self.final_layer = nn.Conv2d(channels + 3, 3, kernel_size=1)

    def forward(self, image, message):
        # first Conv part of Encoder
        image_pre = self.image_pre_layer(image)
        intermediate1 = self.image_first_layer(image_pre)

        if self.is_msg_norm:
            message = message * 2. - 1.
        message = self.msg_fc(message)
        attn = self.attention(image)
        intermediate2 = attn * message.unsqueeze(-1).unsqueeze(-1)
        if self.is_attn_bias:
            attn_bias = self.attention_bias(image)
            intermediate2 = intermediate2 + attn_bias
        intermediate2 = self.message_pre_layer(intermediate2)
        if self.encoder_att_tail:
            intermediate2 = self.attention_tail(intermediate2)
        if self.is_reduce:
            intermediate2 = torch.sum(intermediate2, dim=1, keepdim=True)

        # concatenate
        concat1 = torch.cat([intermediate1, intermediate2], dim=1)

        # second Conv part of Encoder
        intermediate3 = self.after_concat_layer(concat1)

        # skip connection
        concat2 = torch.cat([intermediate3, image], dim=1)

        # last Conv part of Encoder
        output = self.final_layer(concat2)

        return output


class Encoder_Rept(nn.Module):
    '''
    Insert a watermark into an image
    '''

    def __init__(self, H, W, message_length, blocks=4, channels=64, opt={}):
        super(Encoder_Rept, self).__init__()

        self.image_pre_layer = ConvBNRelu(3, channels)
        self.image_first_layer = SENet(channels, channels, blocks=blocks)

        self.message_layer0 = nn.Sequential(
            nn.Linear(message_length, opt['fc_msg_num'])
        )
        self.message_pre_layer = ConvBNRelu(opt['fc_msg_num'], channels)
        self.message_first_layer = SENet(channels, channels, blocks=blocks)

        self.after_concat_layer = ConvBNRelu(2 * channels, channels)

        self.final_layer = nn.Conv2d(channels + 3, 3, kernel_size=1)

    def forward(self, image, message):
        # first Conv part of Encoder
        image_pre = self.image_pre_layer(image)
        intermediate1 = self.image_first_layer(image_pre)

        # message = message * 2. - 1.
        intermediate2 = self.message_layer0(message).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, image.shape[2], image.shape[3])
        intermediate2 = self.message_first_layer(self.message_pre_layer(intermediate2))

        # concatenate
        concat1 = torch.cat([intermediate1, intermediate2], dim=1)

        # second Conv part of Encoder
        intermediate3 = self.after_concat_layer(concat1)

        # skip connection
        concat2 = torch.cat([intermediate3, image], dim=1)

        # last Conv part of Encoder
        output = self.final_layer(concat2)

        return output




class Encoder_Hid(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, H, W, message_length, opt={}):
        super(Encoder_Hid, self).__init__()
        self.H = 256
        self.W = 256
        self.conv_channels = 64
        self.num_blocks = 4

        layers = [ConvBNRelu(3, self.conv_channels)]

        for _ in range(4-1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + 256,
                                             self.conv_channels)

        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)

    def forward(self, image, message):

        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)

        expanded_message = expanded_message.expand(-1,-1, self.H, self.W)
        encoded_image = self.conv_layers(image)
        # concatenate expanded message and image
        concat = torch.cat([expanded_message, encoded_image, image], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)
        return im_w


class Encoder_Haar(nn.Module):
    '''
    Insert a watermark into an image
    '''

    def __init__(self, H, W, message_length, blocks=4, channels=64, opt={}):
        super(Encoder_Haar, self).__init__()
        self.H = H
        self.W = W

        # message_convT_blocks = int(np.log2(H // int(np.sqrt(message_length))))
        message_convT_blocks = 4
        message_se_blocks = max(blocks - message_convT_blocks, 1)

        self.image_pre_layer = ConvBNRelu(3, channels)
        self.image_first_layer = SENet(channels, channels, blocks=blocks)

        self.message_pre_layer_0 = ConvBNRelu(1, channels)
        self.message_pre_layer_1 = ConvPixelShufHaar(channels, 2)
        self.message_pre_layer_2 = SENet(channels, channels, blocks=message_se_blocks)

        self.message_first_layer = SENet(channels, channels, blocks=blocks)

        self.after_concat_layer = ConvBNRelu(2 * channels, channels)

        self.final_layer = nn.Conv2d(channels + 3, 3, kernel_size=1)

    def forward(self, image, message):
        # first Conv part of Encoder
        image_pre = self.image_pre_layer(image)
        intermediate1 = self.image_first_layer(image_pre)

        # Message Processor
        # size = int(np.sqrt(message.shape[1]))
        # message_image = message.view(-1, 1, size, size)
        _, __, h, w = image.shape
        message_image = message.view(-1, 1, h // 16, w // 16)
        message_pre = self.message_pre_layer_0(message_image)
        message_pre = self.message_pre_layer_1(message_pre, image)
        message_pre = self.message_pre_layer_2(message_pre)

        intermediate2 = self.message_first_layer(message_pre)

        # concatenate
        concat1 = torch.cat([intermediate1, intermediate2], dim=1)

        # second Conv part of Encoder
        intermediate3 = self.after_concat_layer(concat1)

        # skip connection
        concat2 = torch.cat([intermediate3, image], dim=1)

        # last Conv part of Encoder
        output = self.final_layer(concat2)

        return output
