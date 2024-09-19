import torch
import torch.nn as nn

class Conv2DBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, padding):
        super(Conv2DBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, padding),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(out_c)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        return self.block(x)


class TransConv2DBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, padding, output_padding=1):
        super(TransConv2DBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel, stride, padding, output_padding=output_padding),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(out_c)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.block(x)


class AutoEncoder(nn.Module):
    def __init__(self, input_shape, conv1_shape=64, conv2_shape=128, init_kernel_size = (4, 6), init_stride=1,
                 init_padding = (1,2), kernel_size=3, stride=2, padding=1, output_padding=0):
        super(AutoEncoder, self).__init__()

        # 321 x 321 x 3
        self.conv1 = Conv2DBlock(input_shape, input_shape, init_kernel_size, init_stride, init_padding) # 320 x 320 x 3
        self.conv2 = Conv2DBlock(input_shape, conv1_shape, kernel_size, stride, padding) # 160 x 160 x 64
        self.conv3 = Conv2DBlock(conv1_shape, conv2_shape, kernel_size, stride, padding) # 80 x 80 x 128


        self.deconv1 = TransConv2DBlock(conv2_shape, conv1_shape, kernel_size, stride, padding) # 160 x 160 x 64
        self.deconv2 = TransConv2DBlock(conv1_shape, input_shape, kernel_size, stride, padding) # 320 x 320 x 3
        self.deconv3 = TransConv2DBlock(input_shape, input_shape, init_kernel_size, init_stride, init_padding,
                                        output_padding=output_padding) # 321 x 321 x 3
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv2(x)
        x2 = self.conv3(x1)  # --> latent space

        y1 = self.deconv1(x2)
        y2 = self.deconv2(y1)
        y3 = self.deconv3(y2)

        y = self.sigmoid(y3)

        return y


class UNet(nn.Module):
    def __init__(self, input_shape, conv1_shape=64, conv2_shape=128, conv3_shape=256, conv4_shape=512,
                 init_kernel_size = (4, 6), init_stride=1, init_padding = (1,2), output_padding=0):
        super(UNet, self).__init__()

        # input --> 321 x 321 x 3
        self.conv1 = Conv2DBlock(input_shape, conv1_shape, init_kernel_size, init_stride, init_padding)
        # 320 x 320 x 64
        self.conv2 = Conv2DBlock(conv1_shape, conv1_shape, 3, 2, 1)
        # 160 x 160 x 64
        self.conv3 = Conv2DBlock(conv1_shape, conv2_shape, 3, 2, 1)
        # 80 x 80 x 128
        self.conv4 = Conv2DBlock(conv2_shape, conv3_shape, 3, 2, 1)
        # 40 x 40 x 256
        self.conv5 = Conv2DBlock(conv3_shape, conv4_shape, 3, 2, 1)
        # 20 x 20 x 512

        self.deconv1 = TransConv2DBlock(conv4_shape, conv3_shape, 3, 2, 1)
        # 40 x 40 x 256
        self.conv6 = Conv2DBlock(conv4_shape, conv3_shape, 3, 1, 1)
        # 40 x 40 x 256

        self.deconv2 = TransConv2DBlock(conv3_shape, conv2_shape, 3, 2, 1)
        # 80 x 80 x 128
        self.conv7 = Conv2DBlock(conv3_shape, conv2_shape, 3, 1, 1)
        # 80 x 80 x 128

        self.deconv3 = TransConv2DBlock(conv2_shape, conv1_shape, 3, 2, 1)
        # 160 x 160 x 64
        self.conv8 = Conv2DBlock(conv2_shape, conv1_shape, 3, 1, 1)
        # 160 x 160 x 64

        self.deconv4 = TransConv2DBlock(conv1_shape, conv1_shape, 3, 2, 1)
        # 320 x 320 x 64
        self.conv9 = Conv2DBlock(conv2_shape, conv1_shape, 3, 1, 1)
        # 320 x 320 x 64

        self.deconv5 = TransConv2DBlock(conv1_shape, conv1_shape, init_kernel_size, init_stride, init_padding,
                                        output_padding=output_padding)
        # 321 x 321 x 64
        self.conv10 = nn.Conv2d(conv1_shape, input_shape, 3, 1, 1)
        # 321 x 321 x 3

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, predict=False):
        # 321 x 321 x 3
        x = self.conv1(x)  # 320 x 320 x 64
        x1 = self.conv2(x)  # 160 x 160 x 64
        x2 = self.conv3(x1)  # 80 x 80 x 128
        x3 = self.conv4(x2)  # 40 x 40 x 256
        x4 = self.conv5(x3)  # 20 x 20 x 512

        y4 = self.deconv1(x4)  # 40 x 40 x 256
        y4 = torch.cat((y4, x3), 1)  # 40 x 40 x 512
        y4 = self.conv6(y4)  # 40 x 40 x 256

        y3 = self.deconv2(y4)  # 80 x 80 x 128
        y3 = torch.cat((y3, x2), 1)  # 80 x 80 x 256
        y3 = self.conv7(y3)  # 80 x 80 x 128

        y2 = self.deconv3(y3)  # 160 x 160 x 64
        y2 = torch.cat((y2, x1), 1)  # 160 x 160 x 128
        y2 = self.conv8(y2)  # 160 x 160 x 64

        y1 = self.deconv4(y2)  # 320 x 320 x 64
        y1 = torch.cat((y1, x), 1)  # 320 x 320 x 128
        y1 = self.conv9(y1)  # 320 x 320 x 64

        y0 = self.deconv5(y1)  # 321 x 321 x 64
        y0 = self.conv10(y0)  # 321 x 321 x 3

        y = self.sigmoid(y0)

        return y


class UNetPlus(nn.Module):
    def __init__(self, input_shape, init_kernel_size = (4, 6), init_stride=1, init_padding = (1,2), output_padding=0):
        super(UNetPlus, self).__init__()

        # input --> 321 x 321 x 3
        self.cnn1 = Conv2DBlock(input_shape, 64, init_kernel_size, init_stride, init_padding)  # 320 x 320 x 64
        self.cnn2 = Conv2DBlock(64, 64, 3, 2, 1)  # 160 x 160 x 64
        self.cnn5 = Conv2DBlock(64, 128, 3, 2, 1)  # 80 x 80 x 128
        self.cnn9 = Conv2DBlock(128, 256, 3, 2, 1)  # 40 x 40 x 256
        self.cnn14 = Conv2DBlock(256, 512, 3, 2, 1)  # 20 x 20 x 512

        # 1st output
        self.dcnn1 = TransConv2DBlock(64, 64, 3, 2, 1)  # 320 x 320 x 64
        self.cnn3 = Conv2DBlock(128, 64, 3, 1, 1)  # 320 x 320 x 64

        self.dcnn2 = TransConv2DBlock(64, 64, init_kernel_size, init_stride, init_padding,
                                      output_padding=output_padding)  # 321 x 321 x 64
        self.cnn4 = nn.Conv2d(64, 3, 3, 1, 1)  # 321 x 321 x 3

        # 2nd output
        self.dcnn3 = TransConv2DBlock(128, 64, 3, 2, 1)  # 160 x 160 x 64
        self.cnn6 = Conv2DBlock(128, 64, 3, 1, 1)  # 160 x 160 x 64

        self.dcnn4 = TransConv2DBlock(64, 64, 3, 2, 1)  # 320 x 320 x 64
        self.cnn7 = Conv2DBlock(128, 64, 3, 1, 1)  # 320 x 320 x 64

        self.dcnn5 = TransConv2DBlock(64, 64, init_kernel_size, init_stride, init_padding,
                                      output_padding=output_padding)  # 321 x 321 x 64
        self.cnn8 = nn.Conv2d(64, 3, 3, 1, 1)  # 321 x 321 x 3

        # 3rd output
        self.dcnn6 = TransConv2DBlock(256, 128, 3, 2, 1)  # 80 x 80 x 128
        self.cnn10 = Conv2DBlock(256, 128, 3, 1, 1)  # 80 x 80 x 128

        self.dcnn7 = TransConv2DBlock(128, 64, 3, 2, 1)  # 160 x 160 x 64
        self.cnn11 = Conv2DBlock(128, 64, 3, 1, 1)  # 160 x 160 x 64

        self.dcnn8 = TransConv2DBlock(64, 64, 3, 2, 1)  # 320 x 320 x 64
        self.cnn12 = Conv2DBlock(128, 64, 3, 1, 1)  # 320 x 320 x 64

        self.dcnn9 = TransConv2DBlock(64, 64, init_kernel_size, init_stride, init_padding,
                                      output_padding=output_padding)  # 321 x 321 x 64
        self.cnn13 = nn.Conv2d(64, 3, 3, 1, 1)  # 321 x 321 x 3

        # Final output
        self.dcnn10 = TransConv2DBlock(512, 256, 3, 2, 1)  # 40 x 40 x 256
        self.cnn15 = Conv2DBlock(512, 256, 3, 1, 1)  # 40 x 40 x 256

        self.dcnn11 = TransConv2DBlock(256, 128, 3, 2, 1)  # 80 x 80 x 128
        self.cnn16 = Conv2DBlock(256, 128, 3, 1, 1)  # 80 x 80 x 128

        self.dcnn12 = TransConv2DBlock(128, 64, 3, 2, 1)  # 160 x 160 x 64
        self.cnn17 = Conv2DBlock(128, 64, 3, 1, 1)  # 160 x 160 x 64

        self.dcnn13 = TransConv2DBlock(64, 64, 3, 2, 1)  # 320 x 320 x 64
        self.cnn18 = Conv2DBlock(128, 64, 3, 1, 1)  # 320 x 320 x 64

        self.dcnn14 = TransConv2DBlock(64, 64, init_kernel_size, init_stride, init_padding,
                                      output_padding=output_padding)  # 321 x 321 x 64
        self.cnn19 = nn.Conv2d(64, 3, 3, 1, 1)  # 321 x 321 x 3

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, predict=False):
        # 321 x 321 x 3
        x = self.cnn1(x)  # 320 x 320 x 64
        x1 = self.cnn2(x)  # 160 x 160 x 64

        y1 = self.dcnn1(x1)
        y1 = torch.cat((y1, x), 1)
        y1 = self.cnn3(y1)
        y1 = self.dcnn2(y1)
        y1 = self.cnn4(y1)
        y10 = self.sigmoid(y1)

        x2 = self.cnn5(x1)

        y2 = self.dcnn3(x2)
        y2 = torch.cat((y2, x1), 1)
        y2 = self.cnn6(y2)
        y2 = self.dcnn4(y2)
        y2 = torch.cat((y2, x), 1)
        y2 = self.cnn7(y2)
        y2 = self.dcnn5(y2)
        y2 = self.cnn8(y2)
        y11 = self.sigmoid(y2)

        x3 = self.cnn9(x2)

        y3 = self.dcnn6(x3)
        y3 = torch.cat((y3, x2), 1)
        y3 = self.cnn10(y3)
        y3 = self.dcnn7(y3)
        y3 = torch.cat((y3, x1), 1)
        y3 = self.cnn11(y3)
        y3 = self.dcnn8(y3)
        y3 = torch.cat((y3, x), 1)
        y3 = self.cnn12(y3)
        y3 = self.dcnn9(y3)
        y3 = self.cnn13(y3)
        y12 = self.sigmoid(y3)

        x4 = self.cnn14(x3)

        y4 = self.dcnn10(x4)
        y4 = torch.cat((y4, x3), 1)
        y4 = self.cnn15(y4)
        y4 = self.dcnn11(y4)
        y4 = torch.cat((y4, x2), 1)
        y4 = self.cnn16(y4)
        y4 = self.dcnn12(y4)
        y4 = torch.cat((y4, x1), 1)
        y4 = self.cnn17(y4)
        y4 = self.dcnn13(y4)
        y4 = torch.cat((y4, x), 1)
        y4 = self.cnn18(y4)
        y4 = self.dcnn14(y4)
        y4 = self.cnn19(y4)
        y13 = self.sigmoid(y4)

        return y13, y12, y11, y10