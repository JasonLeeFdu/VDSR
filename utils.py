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


if __name__ == '__main__':
    '''
    saveCheckpoint('1', 2, 28342, fnCore='model')
    saveCheckpoint('q1',3, 8342, fnCore='model')
    saveCheckpoint('qwett1', 3, 28342, fnCore='model')
    saveCheckpoint('dfsa1', 2, 2, fnCore='model')
    saveCheckpoint('sdf1', 1, 728342, fnCore='model')
    '''
    loadLatestCheckpoint()