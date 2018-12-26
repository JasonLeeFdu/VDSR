# 一个很好的pytorch网站：
# https://www.cnblogs.com/hellcat/p/8477195.html

import torch.nn as nn
import torch
import math
import Configurations as conf
import torch.nn.functional as F

'''
1.注意问题：
1)每一个forward函数都应该有返回值
2)torch的网络层添加方式不同于tf的，后者每一次调用都是新的，前者可能需要借助Sequntial之类的模块辅助
3)*:split the input list-attr into seperate input items; **: split the input dict and organize function input with the key and value
4）当网络输出比较大的时候，请留意初始化，尤其是初始化的方差。
'''




class Inner18Layers(nn.Module): ## OP + forward类型,最为一个自定义nn.Module整体
    def __init__(self):
        super(Inner18Layers, self).__init__()
        self.op = nn.Sequential()
        for i in range(18):
            self.op.add_module('conv_' + str(i),nn.Conv2d(64,64,3,1,1))
            self.op.add_module('relu_'+str(i),nn.ReLU(inplace=True))
    def forward(self, x):
        res = self.op(x)
        return res

## specify module fine-grind类型，充分使用预先定义好的模型
def inner18LayersMod():
    block = nn.Sequential()
    for i in range(18):
        block.add_module('conv_' + str(i), nn.Conv2d(64, 64, 3, 1, 1))
        block.add_module('relu_' + str(i), nn.ReLU(inplace=True))
    return block



class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        self.conv1 = nn.Conv2d(1,64,3,1,1)
        self.convRelu2 = Inner18Layers()#Inner18Layers()   inner18LayersMod
        self.conv3 = nn.Conv2d(64, 1, 3, 1, 1)
        self.reg()

    def forward(self, x):
        conv_1 = F.relu(self.conv1(x))
        lastconv = self.convRelu2(conv_1)
        res = self.conv3(lastconv)
        return res
    def reg(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n) / conf.WEIGHT_INIT_STDDEV_FACTOR)
