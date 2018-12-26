# 一个很好的pytorch网站：
# https://www.cnblogs.com/hellcat/p/8477195.html

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import Configurations as conf
import lmdb


class convLayer(nn.Module):
    def __init__(self,ksz,inc,outc):
        pad = (ksz - 1) / 2
        self.conv = nn.Conv2d(inc,outc,ksz,padding=pad)
        self.relu = nn.ReLU()
    def forward(self, x):
        conv = self.conv(x)
        relu = self.relu(conv)


class VDSR(nn.Module):
    def __init__(self):
        self.conv1 = convLayer(3,1,64)
        #self.conv2 = convLayer(3, 64, 64)
        self.conv3 = convLayer(3, 64, 1)
    def forward(self, x):
        conv_1 = self.conv1(x)
        conv_2 = conv_1
        for i in range(20):
            #conv_2 = self.conv2(conv_2)
            conv2 = convLayer(3,64,64)
            conv_2 = self.conv2(conv_2) # which means different layers with unique param
        conv_3 = self.conv3(conv_2)
        out    = conv_3 + x
        return out



def main():




if __name__ == '__main__':
        main()
