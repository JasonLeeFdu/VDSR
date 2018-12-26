import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import Configurations as conf
import dataManipulator as dm
import network as nt
import lmdb
import os
import numpy as np
import time
import random
import torch.utils.data as data
from PIL import Image
import unbufferOutput
import cv2 as cv
from datetime import datetime
import math
import pickle
import io
import torch
import network as nt
import scipy

def saveCheckpoint(netModel,epoch,iterr,glbiter,fnCore='model'):
    ##net_state = netModel.state_dict()
    res = dict()
    ##res['NetState'] = net_state
    res['NetState'] = netModel
    res['Epoch'] = epoch
    res['Iter']  = iterr
    res['GlobalIter'] = glbiter
    fn  = fnCore + '_' + str(epoch) + '_'+str(iterr)+'.mdl'
    pfn = os.path.join(conf.MODEL_PATH,fn)
    pfnFile = io.open(pfn,mode='wb')
    pickle.dump(res,pfnFile)


def loadSpecificCheckpointNetState1(epoch,iterr,fnCore='model'):
    fn = fnCore + '_' + str(epoch) + '_' + str(iterr) + '.mdl'
    pfn = os.path.join(conf.MODEL_PATH,fn)
    res = pickle.load(pfn)
    net_state = res['NetState']
    globalIter = res['GlobalIter']
    return net_state,globalIter

def loadLatestCheckpoint(fnCore='model'):
    # return net_status epoch iterr
    modelPath = conf.MODEL_PATH
    candidateCpSet = os.listdir(modelPath)
    candidateCpSet = [x for x in candidateCpSet if x.startswith(fnCore) and x.endswith('.mdl')]
    if len(candidateCpSet) == 0:
        return None,0,0,0
    ref            = [x.split('.')[0] for x in candidateCpSet]
    ref1           = [x.split('_')[1] for x in ref]
    ref2           = [x.split('_')[2] for x in ref]
    factor         = 10**len(sorted(ref2, key=lambda k: len(k), reverse=True)[0])
    ref1           = [int(x) for x in ref1]
    ref2           = [int(x) for x in ref2]
    reff           = list(zip(ref1,ref2))
    reff           = [x[0]*factor+x[1] for x in reff]
    idx            = reff.index(max(reff))
    latestCpFn     = candidateCpSet[idx]
    latestCpFnFIO  = io.open(os.path.join(modelPath,latestCpFn),'rb')
    res            = pickle.load(latestCpFnFIO)
    return res['NetState'],res['Epoch'],res['Iter'],res['GlobalIter']


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def gradientClip(netParames,factor):
    for param in netParames:
        param.grad.data = torch.max(torch.FloatTensor([-factor]).expand_as(param).cuda(), param.grad.data)
        param.grad.data = torch.min(torch.FloatTensor([factor]).expand_as(param).cuda(), param.grad.data)

def PSNR_NP_Y255(im1, im2):
    if len(im1.shape) != len(im2.shape):
        print('Unmatched shape!')
        return
    if len(im1.shape) == 3 and im1.shape[2] == 3:
        im1yuv = cv.cvtColor(im1, cv.COLOR_BGR2YCR_CB)
        im1 = im1yuv[:, :, 0]
    if len(im2.shape) == 3 and im2.shape[2] == 3:
        im2yuv = cv.cvtColor(im2, cv.COLOR_BGR2YCR_CB)
        im2 = im2yuv[:, :, 0]
    imdelta = (np.float64(im1) - np.float64(im2))
    rmse = math.sqrt(np.mean(imdelta ** 2))
    psnr = 20 * np.log10(255 / rmse)
    return psnr


def PSNR_PT_Y255(in1,in2):
    #NCHW
    #if
    in1 = torch.squeeze(in1)
    in2 = torch.squeeze(in2)

    if in1.shape != in2.shape:
        return
    imdelta = in1 - in2
    rmse = torch.sqrt(torch.mean(torch.pow(imdelta,2)))
    psnr = 20*torch.log10(255/rmse)
    return psnr

def testPSNR(netModule,testSetPath,scale):
    # read the dirs and get to know the sets
    # we have tsNum ds in dsList
    netModule = netModule.cuda()
    tsNames = os.listdir(testSetPath)
    tsNames.sort()
    tsNum   = len(tsNames)
    resDict = dict()
    transformer = transforms.Compose([
        transforms.ToTensor()
    ])
    # load the datasets
    ds = datasets.ImageFolder(os.path.join(testSetPath),transform=transformer)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)
    dsLoader = dataloader
    sumList = np.zeros(tsNum)
    numList = np.zeros(tsNum)

    for idxTestSet in range(tsNum):
        tsName = tsNames[idxTestSet]
        sum = 0.0;counter = 0;

        for _, smpl in enumerate(dsLoader, 1):
            y = np.array(smpl[0])
            numList[smpl[1]] += 1
            y = np.squeeze(y)
            y = np.transpose(y,[1,2,0])
            h  = y.shape[0]; w = y.shape[1];
            if len(y.shape)  == 3:
                c = 3
            else:
                c = 1
            y = np.minimum(np.maximum(0,np.round(y*255)),255)
            y = y.astype(np.uint8)  # rgb
            y = y[:,:,[2,1,0]]      # bgr
            # Y
            if c == 3:
                yYUV = cv.cvtColor(y, cv.COLOR_BGR2YCR_CB)
                yY = yYUV[:, :, 0]
            else:
                yY = y

            # reshape
            xY = scipy.misc.imresize(yY, (int(h / scale), int(w / scale)),
                                             interp='bicubic', mode=None)
            xY = scipy.misc.imresize(xY, (int(h), int(w)),
                                             interp='bicubic', mode=None)

            inxY = torch.Tensor(xY/255)
            inxY = torch.unsqueeze(torch.unsqueeze(inxY,0),0)

            inxY = inxY.cuda()
            predY = netModule(inxY) + inxY

            predY = torch.max(torch.FloatTensor([0.0]).expand_as(predY).cuda(), predY)
            predY = torch.min(torch.FloatTensor([1.0]).expand_as(predY).cuda(), predY)

            predY = torch.round(predY * 255)


            inY = torch.Tensor(yY)
            inY = inY.cuda()
            res = PSNR_PT_Y255(predY,inY).detach().cpu().numpy()



            predYNp   = predY.detach().cpu().numpy();predYNp = predYNp.astype(np.uint8);predYNp = predYNp.squeeze();
            inYNp     = inY.detach().cpu().numpy();inYNp = inYNp.astype(np.uint8);inYNp = inYNp.squeeze();
            cv.imshow('predYNp'+str(res),predYNp);cv.imshow('inputa',xY);cv.imshow('inY',inYNp);cv.waitKey(0);cv.destroyAllWindows()
            numList[smpl[1]] +=  np.array(res)  



            #psnr = PSNR_PT_Y255(inx.iny)
            #sum
    z = 11

    # Do the test

    # return  the results of the psnr per test









if __name__ == '__main__':
    '''
    saveCheckpoint('1', 2, 28342, fnCore='model')
    saveCheckpoint('q1',3, 8342, fnCore='model')
    saveCheckpoint('qwett1', 3, 28342, fnCore='model')
    saveCheckpoint('dfsa1', 2, 2, fnCore='model')
    saveCheckpoint('sdf1', 1, 728342, fnCore='model')
    '''
    #testPSNR(nt.VDSR(), conf.TEST_SET_PATH)

    # NCHW

    net = nt.VDSR()
    net_stat, epoch, iterr, globalStep = loadLatestCheckpoint()
    if net_stat is None:
        print('No previous model found, start training now')
    else:
        net.load_state_dict(net_stat)
        print('The lastest version is Epoch %d, iter is %d，GLoablStep: %d' % (epoch, iterr, globalStep))
        # globalStep += 1

    testPSNR(net, conf.TEST_SET_PATH,4)
    a  = 1
























    '''
    MAX = 80000

    inx = torch.rand([1,77])
    lbl = inx * 8.932 + 0.4

    inx = torch.FloatTensor(inx).cuda()
    lbl = torch.FloatTensor(lbl).cuda()

    w   = torch.nn.Parameter(torch.FloatTensor([1.0]).cuda(),requires_grad=True)
    h   = torch.nn.Parameter(torch.FloatTensor([2.0]).cuda(),requires_grad=True)

    optimizer = optim.SGD([w,h], lr=0.001)
    for i in range(MAX):
        optimizer.zero_grad()
        y = w.matmul(inx) + h
        loss = torch.mean((lbl - y).pow(2))
        loss.backward()
        if i%1 == 0:
            print('step:',str(i))
            print('W:')
            print(w)
            print('h:')
            H = h.detach()
            H.data.add_(100)
            print(h)
            print('H:')
            print(H)
            print('Loss:')
            print(loss)
            print('----------------------------');print('')
        optimizer.step()
        s  = 1
    '''
