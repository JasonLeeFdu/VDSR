# 一个很好的pytorch网站：
# https://www.cnblogs.com/hellcat/p/8477195.html

import torch.nn as nn
import torch
'''
1.注意问题：
1)每一个forward函数都应该有返回值
2)torch的网络层添加方式不同于tf的，后者每一次调用都是新的，前者可能需要借助Sequntial之类的模块辅助
3)*:split the input list-attr into seperate input items; **: split the input dict and organize function input with the key and value

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


class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        self.conv1 = convLayer(3,1,64)
        self.middleconvs = self.repeatLayers(convLayer,18,3,64,64)
        self.conv3 = convLayer(3, 64, 1)


    def repeatLayers(self,module,repeatTimes,ksz,inc,outc):
        layers= list()
        for i in range(repeatTimes):
            layers.append(module(ksz,inc,outc))
        return nn.Sequential(*layers)



    def forward(self, x):
        conv_1 = self.conv1(x)
        conv_2 = self.middleconvs(conv_1) # which means different layers with unique param
        conv_3 = self.conv3(conv_2)
        out = torch.add(conv_3 , x)
        return out






# def main():
#
#
#
#
# if __name__ == '__main__':
#         main()
