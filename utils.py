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


def testPSNR(netModule,testSetPath):
    # read the dirs and get to know the sets
    # we have tsNum ds in dsList
    tsNames = os.listdir(testSetPath)
    tsNum   = len(tsNames)
    dsLoaderList  = list()
    resDict = dict()
    # load the datasets
    for i in range(tsNum):
        ds = datasets.ImageFolder(os.path.join(testSetPath,tsNames[i]))
        dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)
        dsLoaderList.append(dataloader)

    for idxTestSet in range(tsNum):
        a = 1

    # Do the test

    # return  the results of the psnr per test


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
    im1 = np.array([1.0,2,3,4,5,6,7,8,9])
    im1 = np.reshape(im1,[1,1,3,3])
    im2 = np.array([1.0,2,3,4,5,6,5,6,9])
    im2 = np.reshape(im2,[1,1,3,3])
    im1T = torch.Tensor(im1)
    im2T = torch.Tensor(im2)

    b = PSNR_PT_Y255(im1T, im2T)
    a  = 1

