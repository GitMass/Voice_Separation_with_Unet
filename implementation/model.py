# Architecture of the voice-separator Unet

import torch.nn as nn
import torch.nn.functional as F




# Encoder block
class EncoderConv2DBlock(nn.Module): 

    # init
    def __init__(self, 
                 in_ch, 
                 out_ch, 
                 kernel_size,
                 stride,
                 padding,
                 ReLU_leak
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
                 kernel_size,
                 stride,
                 padding,
                 output_padding,
                 dropout
                 ):
        super().__init__()

        # deconv, for good size management always use padding = (kernelsize-1)/2, bias=false because we are using batch_norm
        self.deconv2d = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False)

        # batch_norm
        self.batch_norm = nn.BatchNorm2d(out_ch)

        # activation function : leakyReLU 0.2
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