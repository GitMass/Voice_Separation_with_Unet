# Architecture of the voice-separator Unet

import torch
import torch.nn as nn




# Encoder block
class EncoderConv2DBlock(nn.Module): 

    # init
    def __init__(self, 
                 in_ch, 
                 out_ch, 
                 kernel_size=5,
                 stride=2,          # to downsample by half
                 padding=2,         # for good size management always use padding = (kernelsize-1)/2, it keeps the halving exact
                 ReLU_leak=0.2
                 ):
        super().__init__()

        # 2D conv, for good size management always use padding = (kernelsize-1)/2, bias=false because we are using batch_norm
        self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        # batch norm
        self.batch_norm = nn.BatchNorm2d(out_ch)

        # activation function : leakyReLU 0.2
        self.activation = nn.LeakyReLU(ReLU_leak)

    # forward
    def forward(self, x):
        x = self.conv2d(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x
    



# Decoder block
class DecoderDeconv2DBlock(nn.Module):

    # init
    def __init__(self, 
                 in_ch,
                 out_ch,
                 dropout=0,
                 kernel_size=5,         
                 stride=2,              # to downsample by half
                 padding=2,             # for good size management always use padding = (kernelsize-1)/2, it keeps the halving exact
                 output_padding=1,      # To perfectly reconstruct the same dimensions from the Encoder
                 activation='relu'
                 ):
        super().__init__()

        # deconv, for good size management always use padding = (kernelsize-1)/2, bias=false because we are using batch_norm
        self.deconv2d = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False)

        # batch_norm, if last layer (sigmoid) we usually do not use batch norm
        if activation == 'sigmoid':
            self.dropout = nn.Identity()
        else :
            self.batch_norm = nn.BatchNorm2d(out_ch)
        

        # activation function : leakyReLU 0.2 (sigmoid if last one)
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else :
            self.activation = nn.ReLU()


        # dropout
        if dropout > 0 :
            self.dropout = nn.Dropout(dropout)
        else :
            self.dropout = nn.Identity()

    # forward
    def forward(self, x) :
        x = self.deconv2d(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
    



# Unet
class UnetSeparator(nn.Module) : 

    # init
    def __init__(self):
        super().__init__()

        in_channels = 1
        base_channels = 16

        # 1. Encoder (6 EncoderConv2DBlock blocks)
        self.down_blocks = nn.ModuleList([
            EncoderConv2DBlock(in_channels, base_channels),                 # 1 → 16
            EncoderConv2DBlock(base_channels, base_channels * 2),           # 16 → 32
            EncoderConv2DBlock(base_channels * 2, base_channels * 4),       # 32 → 64
            EncoderConv2DBlock(base_channels * 4, base_channels * 8),       # 64 → 128
            EncoderConv2DBlock(base_channels * 8, base_channels * 16),      # 128 → 256
            EncoderConv2DBlock(base_channels * 16, base_channels * 32),     # 256 → 512
        ])

        # 2. Decoder (6 EncoderConv2DBlock blocks), dropout in the first 3 layers
        self.up_blocks = nn.ModuleList([
            DecoderDeconv2DBlock(base_channels * 32, base_channels * 16, dropout=0.5),      # 512 → 256
            DecoderDeconv2DBlock(base_channels * 16*2, base_channels * 8, dropout=0.5),       # 256+256 → 128
            DecoderDeconv2DBlock(base_channels * 8*2, base_channels * 4, dropout=0.5),        # 128+128 → 64
            DecoderDeconv2DBlock(base_channels * 4*2, base_channels * 2),                     # 64+64 → 32
            DecoderDeconv2DBlock(base_channels * 2*2, base_channels * 1),                     # 32+32 → 16
            DecoderDeconv2DBlock(base_channels * 1*2, in_channels, activation='sigmoid'),     # 16+16 → 1
        ])

    # forward
    def forward(self, x):

        # encoder 
        e0 = self.down_blocks[0](x)
        e1 = self.down_blocks[1](e0)
        e2 = self.down_blocks[2](e1)
        e3 = self.down_blocks[3](e2)
        e4 = self.down_blocks[4](e3)
        e5 = self.down_blocks[5](e4)

        # encoder 
        d0 = self.up_blocks[0](e5)
        d1 = self.up_blocks[1](torch.cat([d0, e4], dim=1))
        d2 = self.up_blocks[2](torch.cat([d1, e3], dim=1))
        d3 = self.up_blocks[3](torch.cat([d2, e2], dim=1))
        d4 = self.up_blocks[4](torch.cat([d3, e1], dim=1))
        mask = self.up_blocks[5](torch.cat([d4, e0], dim=1))

        # multiplication
        output = mask*x

        return output

