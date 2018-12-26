

import torch
from datagen import MyDataset

trainset = MyDataset(path_file=pathFile,list_file=trainList,numJoints = 6,type=False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)
testset = MyDataset(path_file=pathFile,list_file=testList,numJoints = 6,type=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=8)
1
2
3
4
5
6
7
以下是定义class MyDataset文件datagen.py， 其中有__init__(self, path_file, list_file,numJoints,type)、__getitem__(self, idx)、__len__(self)三个函数，__getitem__返回一个(22,256,256)的输入和一个(6,256,256)的标签。
'''
Load data
'''

import numpy as np
from PIL import Image
#import cv2

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

class MyDataset(data.Dataset):

    def __init__(self, path_file, list_file,numJoints,type):
        '''
        Args:
          path_file: (str) heatmap and optical file location
          list_file: (str) path to index file.
          numJoints: (int) number of joints
          type: (boolean) use pose flow(true) or optical flow(false)
        '''

        self.numJoints = numJoints

        # read heatmap and optical path
        with open(path_file) as f:
            paths = f.readlines()

        for path in paths:
            splited = path.strip().split()
            if splited[0]=='resPath':
                self.resPath = splited[1]
            elif splited[0]=='gtPath':
                self.gtPath = splited[1]
            elif splited[0]=='opticalFlowPath':
                self.opticalFlowPath = splited[1]
            elif splited[0]=='poseFlowPath':
                self.poseFlowPath = splited[1]
        if type:
            self.flowPath = self.poseFlowPath
        else:
            self.flowPath = self.opticalFlowPath


        #read list
        with open(list_file) as f:
            self.list = f.readlines()
            self.num_samples = len(self.list)

def __getitem__(self, idx):
    '''
    load heatmaps and optical flow and encode it to a 22 channels input and 6 channels output
    :param idx: (int) image index
    :return:
        input: a 22 channel input which integrate 2 optical flow and heatmaps of 3 image
        output: the ground truth

    '''

    input = []
    output = []
    # load heatmaps of 3 image
    for im in range(3):
        for map in range(6):
            curResPath = self.resPath + self.list[idx].rstrip('\n') + str(im + 1) + '/' + str(map + 1) + '.bmp'
            heatmap = Image.open(curResPath)
            heatmap.load()
            heatmap = np.asarray(heatmap, dtype='float') / 255
            input.append(heatmap)
    # load 2 flow
    for flow in range(2):
        curFlowXPath = self.flowPath + self.list[idx].rstrip('\n') + 'flowx/' + str(flow + 1) + '.jpg'
        flowX = Image.open(curFlowXPath)
        flowX.load()
        flowX = np.asarray(flowX, dtype='float')
        curFlowYPath = self.flowPath + self.list[idx].rstrip('\n') + 'flowy/' + str(flow + 1) + '.jpg'
        flowY = Image.open(curFlowYPath)
        flowY.load()
        flowY = np.asarray(flowY, dtype='float')
        input.append(flowX)
        input.append(flowY)
    # load groundtruth
    for map in range(6):
        curgtPath = self.resPath + self.list[idx].rstrip('\n') + str(2) + '/' + str(map + 1) + '.bmp'
        heatmap = Image.open(curResPath)
        heatmap.load()
        heatmap = np.asarray(heatmap, dtype='float') / 255
        output.append(heatmap)

    input = torch.Tensor(input)
    output = torch.Tensor(output)

    return input,output



def __len__(self):
    return self.num_samples


'''
---------------------
作者：gyguo95
来源：CSDN
原文：https://blog.csdn.net/gyguo95/article/details/78821520
版权声明：本文为博主原创文章，转载请附上博文链接！
'''

