"""
Pytorch implementation of SegNet (https://arxiv.org/pdf/1511.00561.pdf)
"""

from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.models as models
import pprint

F = nn.functional
DEBUG = False


vgg16_dims = [
                    (64, 64, 'M'),                                # Stage - 1
                    (128, 128, 'M'),                              # Stage - 2
                    (256, 256, 256,'M'),                          # Stage - 3
                    (512, 512, 512, 'M'),                         # Stage - 4
                    (512, 512, 512, 'M')                          # Stage - 5
            ]

decoder_dims = [
                    ('U', 512, 512, 512),                         # Stage - 5
                    ('U', 512, 512, 512),                         # Stage - 4
                    ('U', 256, 256, 256),                         # Stage - 3
                    ('U', 128, 128),                              # Stage - 2
                    ('U', 64, 64)                                 # Stage - 1
                ]


class SegNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SegNet, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.num_channels = input_channels

        #self.vgg16 = models.vgg16(pretrained=True)


        # Encoder layers
        self.encoder_conv00 = nn.Conv2d(self.input_channels,64,3,padding=1)
        self.encoder_conv01 = nn.Conv2d(64,64,3,padding=1)
        self.encoder_conv10 = nn.Conv2d(64,128,3,padding=1)
        self.encoder_conv11 = nn.Conv2d(128,128,3,padding=1)
        self.encoder_conv20 = nn.Conv2d(128,256,3,padding=1)
        self.encoder_conv21 = nn.Conv2d(256,256,3,padding=1)
        self.encoder_conv22 = nn.Conv2d(256,256,3,padding=1)
        self.normal64 = nn.BatchNorm2d(64)
        self.normal128 = nn.BatchNorm2d(128)
        self.normal256 = nn.BatchNorm2d(256)

        self.decoder_convtr22 = nn.ConvTranspose2d(256,256,3,padding=1)
        self.decoder_convtr21 = nn.ConvTranspose2d(256,256,3,padding=1)
        self.decoder_convtr20 = nn.ConvTranspose2d(256,128,3,padding=1)
        self.decoder_convtr11 = nn.ConvTranspose2d(128,128,3,padding=1)
        self.decoder_convtr10 = nn.ConvTranspose2d(128,64,3,padding=1)
        self.decoder_convtr01 = nn.ConvTranspose2d(64,64,3,padding=1)
        self.decoder_convtr00 = nn.ConvTranspose2d(64,self.output_channels,3,padding=1)

       

    def forward(self, input_img):
        """
        Forward pass `input_img` through the network
        """

        # Encoder
        dim_0 = input_img.size()
        x = self.encoder_conv00(input_img)
        x = self.normal64(x)
        x = F.relu(x)
        x = self.encoder_conv01(x)
        x = self.normal64(x)
        x = F.relu(x)  
        x_0, indices_0 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        dim_1 = x_0.size()
        x = self.encoder_conv10(x_0)
        x = self.normal128(x)
        x = F.relu(x)
        x = self.encoder_conv11(x)
        x = self.normal128(x)
        x = F.relu(x)  
        x_1, indices_1 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        dim_2 = x_1.size()
        x = self.encoder_conv20(x_1)
        x = self.normal256(x)
        x = F.relu(x)
        x = self.encoder_conv21(x)
        x = self.normal256(x)
        x = F.relu(x)  
        x = self.encoder_conv22(x)
        x = self.normal256(x)
        x = F.relu(x)
        x_2, indices_2 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)



        # Decoder Stage - 3

        x_2d = F.max_unpool2d(x_2, indices_2, kernel_size=2, stride=2, output_size=dim_2)
        #x_2d = F.max_unpool2d(x_2, indices_2, kernel_size=2, stride=2, output_size=dim_2)
        x = F.relu(self.decoder_convtr22(x_2d))
        x = self.normal256(x)
        x = F.relu(self.decoder_convtr21(x))
        x = self.normal256(x)
        x = F.relu(self.decoder_convtr20(x))
        x = self.normal128(x)

        x_1d = F.max_unpool2d(x, indices_1, kernel_size=2, stride=2, output_size=dim_1)
        x = F.relu(self.decoder_convtr11(x_1d))
        x = self.normal128(x)
        x = F.relu(self.decoder_convtr10(x))
        x = self.normal64(x)


        x_0d = F.max_unpool2d(x, indices_0, kernel_size=2, stride=2, output_size=dim_0)
        x = F.relu(self.decoder_convtr01(x_0d))
        x = self.normal64(x)
        x_00d = F.relu(self.decoder_convtr00(x))
        #x_22d = F.relu(self.decoder_convtr_22(x_2d))
        #x_21d = F.relu(self.decoder_convtr_21(x_22d))
        #x_20d = F.relu(self.decoder_convtr_20(x_21d))
        #dim_2d = x_20d.size()

        # Decoder Stage - 2
        '''x_1d = F.max_unpool2d(x_20d, indices_1, kernel_size=2, stride=2, output_size=dim_1)
        x_11d = F.relu(self.decoder_convtr_11(x_1d))
        x_10d = F.relu(self.decoder_convtr_10(x_11d))
        dim_1d = x_10d.size()

        # Decoder Stage - 1
        x_0d = F.max_unpool2d(x_10d, indices_0, kernel_size=2, stride=2, output_size=dim_0)
        x_01d = F.relu(self.decoder_convtr_01(x_0d))
        x_00d = self.decoder_convtr_00(x_01d)
        dim_0d = x_00d.size()'''

        x_softmax = F.softmax(x_00d, dim=1)


        if DEBUG:
            print("dim_0: {}".format(dim_0))
            print("dim_1: {}".format(dim_1))
            print("dim_2: {}".format(dim_2))
            print("dim_3: {}".format(dim_3))
            print("dim_4: {}".format(dim_4))

            print("dim_d: {}".format(dim_d))
            print("dim_4d: {}".format(dim_4d))
            print("dim_3d: {}".format(dim_3d))
            print("dim_2d: {}".format(dim_2d))
            print("dim_1d: {}".format(dim_1d))
            print("dim_0d: {}".format(dim_0d))


        return x_00d, x_softmax


    """def init_vgg_weigts(self):
        assert self.encoder_conv_00[0].weight.size() == self.vgg16.features[0].weight.size()
        self.encoder_conv_00[0].weight.data = self.vgg16.features[0].weight.data
        assert self.encoder_conv_00[0].bias.size() == self.vgg16.features[0].bias.size()
        self.encoder_conv_00[0].bias.data = self.vgg16.features[0].bias.data

        assert self.encoder_conv_01[0].weight.size() == self.vgg16.features[2].weight.size()
        self.encoder_conv_01[0].weight.data = self.vgg16.features[2].weight.data
        assert self.encoder_conv_01[0].bias.size() == self.vgg16.features[2].bias.size()
        self.encoder_conv_01[0].bias.data = self.vgg16.features[2].bias.data

        assert self.encoder_conv_10[0].weight.size() == self.vgg16.features[5].weight.size()
        self.encoder_conv_10[0].weight.data = self.vgg16.features[5].weight.data
        assert self.encoder_conv_10[0].bias.size() == self.vgg16.features[5].bias.size()
        self.encoder_conv_10[0].bias.data = self.vgg16.features[5].bias.data

        assert self.encoder_conv_11[0].weight.size() == self.vgg16.features[7].weight.size()
        self.encoder_conv_11[0].weight.data = self.vgg16.features[7].weight.data
        assert self.encoder_conv_11[0].bias.size() == self.vgg16.features[7].bias.size()
        self.encoder_conv_11[0].bias.data = self.vgg16.features[7].bias.data

        assert self.encoder_conv_20[0].weight.size() == self.vgg16.features[10].weight.size()
        self.encoder_conv_20[0].weight.data = self.vgg16.features[10].weight.data
        assert self.encoder_conv_20[0].bias.size() == self.vgg16.features[10].bias.size()
        self.encoder_conv_20[0].bias.data = self.vgg16.features[10].bias.data

        assert self.encoder_conv_21[0].weight.size() == self.vgg16.features[12].weight.size()
        self.encoder_conv_21[0].weight.data = self.vgg16.features[12].weight.data
        assert self.encoder_conv_21[0].bias.size() == self.vgg16.features[12].bias.size()
        self.encoder_conv_21[0].bias.data = self.vgg16.features[12].bias.data

        assert self.encoder_conv_22[0].weight.size() == self.vgg16.features[14].weight.size()
        self.encoder_conv_22[0].weight.data = self.vgg16.features[14].weight.data
        assert self.encoder_conv_22[0].bias.size() == self.vgg16.features[14].bias.size()
        self.encoder_conv_22[0].bias.data = self.vgg16.features[14].bias.data

        '''assert self.encoder_conv_30[0].weight.size() == self.vgg16.features[17].weight.size()
        self.encoder_conv_30[0].weight.data = self.vgg16.features[17].weight.data
        assert self.encoder_conv_30[0].bias.size() == self.vgg16.features[17].bias.size()
        self.encoder_conv_30[0].bias.data = self.vgg16.features[17].bias.data

        assert self.encoder_conv_31[0].weight.size() == self.vgg16.features[19].weight.size()
        self.encoder_conv_31[0].weight.data = self.vgg16.features[19].weight.data
        assert self.encoder_conv_31[0].bias.size() == self.vgg16.features[19].bias.size()
        self.encoder_conv_31[0].bias.data = self.vgg16.features[19].bias.data

        assert self.encoder_conv_32[0].weight.size() == self.vgg16.features[21].weight.size()
        self.encoder_conv_32[0].weight.data = self.vgg16.features[21].weight.data
        assert self.encoder_conv_32[0].bias.size() == self.vgg16.features[21].bias.size()
        self.encoder_conv_32[0].bias.data = self.vgg16.features[21].bias.data'''

        '''assert self.encoder_conv_40[0].weight.size() == self.vgg16.features[24].weight.size()
        self.encoder_conv_40[0].weight.data = self.vgg16.features[24].weight.data
        assert self.encoder_conv_40[0].bias.size() == self.vgg16.features[24].bias.size()
        self.encoder_conv_40[0].bias.data = self.vgg16.features[24].bias.data

        assert self.encoder_conv_41[0].weight.size() == self.vgg16.features[26].weight.size()
        self.encoder_conv_41[0].weight.data = self.vgg16.features[26].weight.data
        assert self.encoder_conv_41[0].bias.size() == self.vgg16.features[26].bias.size()
        self.encoder_conv_41[0].bias.data = self.vgg16.features[26].bias.data

        assert self.encoder_conv_42[0].weight.size() == self.vgg16.features[28].weight.size()
        self.encoder_conv_42[0].weight.data = self.vgg16.features[28].weight.data
        assert self.encoder_conv_42[0].bias.size() == self.vgg16.features[28].bias.size()
        self.encoder_conv_42[0].bias.data = self.vgg16.features[28].bias.data'''"""
