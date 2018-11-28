# 一个很好的pytorch网站：
# https://www.cnblogs.com/hellcat/p/8477195.html

import torch.nn as nn
import torch
import math
import Configurations as conf
'''
1.注意问题：
1)每一个forward函数都应该有返回值
2)torch的网络层添加方式不同于tf的，后者每一次调用都是新的，前者可能需要借助Sequntial之类的模块辅助
3)*:split the input list-attr into seperate input items; **: split the input dict and organize function input with the key and value
4）当网络输出比较大的时候，请留意初始化，尤其是初始化的方差。
'''


class convLayer(nn.Module):
    def __init__(self,ksz,inc,outc):
        super(convLayer, self).__init__()
        pad = int((ksz - 1) / 2)  # The pad should be an integer
        self.conv = nn.Conv2d(inc,outc,ksz,padding=pad)
        self.relu = nn.ReLU()
    def forward(self, x):
        conv = self.conv(x)
        relu = self.relu(conv)
        return relu

'''

class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        self.conv1 = convLayer(3,1,64)
        #self.middleconvs = self.repeatConvLayers()
        self.conv3 = convLayer(3, 64, 1)


    def repeatConvLayers(self):
        block = nn.Sequential()
        for i in range(18):
            block.add_module('middle'+str(i), convLayer(3, 64, 64))
        return block



    def forward(self, x):
        conv_1 = self.conv1(x)
        #conv_2 = self.middleconvs(conv_1) # which means different layers with unique param
        conv_3 = self.conv3(conv_1)
        out = torch.add(conv_3 , x)
        return out

'''

VDSR = nn.Sequential()
VDSR.add_module('conv1',nn.Conv2d(1,64,3,1,1))
VDSR.add_module('conv2',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv3',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv4',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv5',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv6',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv7',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv8',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv9',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv10',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv11',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv12',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv13',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv14',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv15',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv16',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv17',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv18',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv19',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv20',nn.Conv2d(64,1,3,1,1))

for m in VDSR.modules():
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n)/conf.WEIGHT_INIT_STDDEV_FACTOR)


# def main():
#
#
#
#
# if __name__ == '__main__':
#         main()

'''
VDSR = nn.Sequential()
VDSR.add_module('conv1',nn.Conv2d(1,64,3,1,1))
VDSR.add_module('conv2',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv3',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv4',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv5',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv6',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv7',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv8',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv9',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv10',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv11',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv12',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv13',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv14',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv15',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv16',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv17',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv18',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv19',nn.Conv2d(64,64,3,1,1))
VDSR.add_module('conv20',nn.Conv2d(64,1,3,1,1))


'''