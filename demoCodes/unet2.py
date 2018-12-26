UNet128(nn.Module):


def __init__(self, in_channel):
    super(UNet128, self).__init__()

    # 128
    self.down3 = StackEncoder(in_channel, 128, kernel_size=3)  # 64
    self.down4 = StackEncoder(128, 256, kernel_size=3)  # 32
    self.down5 = StackEncoder(256, 512, kernel_size=3)  # 16
    self.down6 = StackEncoder(512, 1024, kernel_size=3)  # 8

    self.center = nn.Sequential(
        ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1),
    )

    # 8
    # x_big_channels, x_channels, y_channels
    self.up6 = StackDecoder(1024, 1024, 512, kernel_size=3)  # 16
    self.up5 = StackDecoder(512, 512, 256, kernel_size=3)  # 32
    self.up4 = StackDecoder(256, 256, 128, kernel_size=3)  # 64
    self.up3 = StackDecoder(128, 128, 64, kernel_size=3)  # 128
    self.classify = nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1,
                              bias=True)  # 1*1 kernel, 0 padding, 1 stride 输出size当然和输入保持一致


def forward(self, x):
    out = x  #
    #        print('x    ',x.size())
    down3, out = self.down3(out)  #
    #        print('down3',down3.size())
    down4, out = self.down4(out)  #
    #        print('down4',down4.size())
    down5, out = self.down5(out)  #
    #        print('down5',down5.size())
    down6, out = self.down6(out)  #
    #        print('down6',down6.size())
    #        print('out  ',out.size())

    out = self.center(out)
    #        print('center',out.size())
    out = self.up6(down6, out)
    #        print('up6',out.size())        #特征融合：down6和out
    out = self.up5(down5, out)
    #        print('up5',out.size())
    out = self.up4(down4, out)
    #        print('up4',out.size())
    out = self.up3(down3, out)  # down3尺寸和x一样，保证输出尺寸和原图相同
    #        print('up3',out.size())
    out = self.classify(out)
    #        print('classify',out.size())
    out = torch.squeeze(out, dim=1)

    return out