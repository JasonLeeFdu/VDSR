#!/usr/bin/env python -u
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
import os
import io
import numpy as np
import time
import random
import torch.utils.data as data
from PIL import Image
import cv2 as cv
import scipy.misc
import matplotlib.pyplot as plt

IDCounter  = 0
ksMod      = '%010d'
TOTAL_NUM  = 10000
TEST_NUM   = 120


class SRDB291(data.Dataset):
    # properties
    # self.txn -- The handle to read the dataset
    # self.len -- The total length of the dataset entries
    def __del__(self):
       a = 1

    def __init__(self):
        self.txn = None
        self.env = None
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            ]
        )


        dsFile = os.path.join(conf.DATASET_DIR,conf.DATASET_FN)
        if not os.path.exists(dsFile):
            #build up the dataset
            #preparations
            dsFns = os.listdir(conf.SRDS291_PATH)
            ksMod = '%010d'
            psz     = conf.PATCH_SIZE
            pID     = 0
            print('==================  Processing the dataset  =======================',end='')
            with lmdb.open(os.path.join(conf.DATASET_DIR, conf.DATASET_FN), map_size=1099511627776) as env:
                txn = env.begin(write=True) # return the transaction
                fnCounter = 0
                byteBuff = io.BytesIO()
                for fn in dsFns:
                    fnCounter += 1
                    print('')
                    print('NO.'+str(fnCounter)+'------《'+fn+'》'+':')
                    fpn = os.path.join(conf.SRDS291_PATH,fn)
                    img = Image.open(fpn)
                    innerCounter = 0
                    pstrd = random.randint(int(psz / 3.5) - 2, int(psz / 3.5) + 2)
                    w,h = img.size
                    li = list(range(int(w / pstrd)))
                    stepX = [x*pstrd for x in li if (x*pstrd)<=(w-psz) ]
                    li = list(range(int(h / pstrd)))
                    stepY = [x * pstrd for x in li if (x * pstrd) <= (h - psz)]
                    flip  = [0,1]
                    rotate = [0,1,2,3]
                    ### for-for-for-for loop, flip and rotation flag
                    for xs in stepX:
                        for ys in stepY:
                            # try to crop xs:xs+pstrd-1,ys:ys+pstrd-1.(mlv)
                            patch = img.crop((xs,ys,xs+psz,ys+psz))         # the PIL image's crop is just the same as that of 'range'
                            for fl in flip:
                                if fl == 0:
                                    patchf = patch
                                elif fl == 1:
                                    patchf = patch.transpose(Image.FLIP_LEFT_RIGHT)
                                for rt in rotate:
                                    if rt == 0:
                                        patchfr = patchf
                                    elif rt == 1:
                                        patchfr = patchf.transpose(Image.ROTATE_90)
                                    elif rt == 2:
                                        patchfr = patchf.transpose(Image.ROTATE_180)
                                    elif rt == 3:
                                        patchfr = patchf.transpose(Image.ROTATE_270)
                                    ## to write the patchfr into the dataset file



                                    # Making examples
                                    npMat = np.array(patchfr)
                                    cvMat = cv.cvtColor(npMat,cv.COLOR_RGB2BGR)
                                    cvMatYUV = cv.cvtColor(cvMat,cv.COLOR_BGR2YCR_CB)
                                    cvMatY   = cvMatYUV[:,:,0]

                                    cvMatY_X_2 = scipy.misc.imresize(cvMatY, (int(psz / 2), int(psz / 2)),
                                                                     interp='bicubic', mode=None)
                                    cvMatY_X_2 = scipy.misc.imresize(cvMatY_X_2, (int(psz ), int(psz )),
                                                                     interp='bicubic', mode=None)

                                    cvMatY_X_3 = scipy.misc.imresize(cvMatY, (int(psz / 3), int(psz / 3)),
                                                                     interp='bicubic', mode=None)
                                    cvMatY_X_3 = scipy.misc.imresize(cvMatY_X_3, (int(psz ), int(psz )),
                                                                     interp='bicubic', mode=None)

                                    cvMatY_X_4 = scipy.misc.imresize(cvMatY, (int(psz / 4), int(psz / 4)),
                                                                     interp='bicubic', mode=None)
                                    cvMatY_X_4 = scipy.misc.imresize(cvMatY_X_4, (int(psz ), int(psz )),
                                                                     interp='bicubic', mode=None)

                                    cvMatY_     = np.expand_dims(cvMatY,axis=2)
                                    cvMatY_X_2_ = np.expand_dims(cvMatY_X_2,axis=2)
                                    cvMatY_X_3_ = np.expand_dims(cvMatY_X_3, axis=2)
                                    cvMatY_X_4_ = np.expand_dims(cvMatY_X_4, axis=2)
                                    ###$$$


                                    cvMatsSample2 = np.concatenate((cvMatY_,cvMatY_X_2_),axis=2) # PIL LA mode
                                    cvMatsSample3 = np.concatenate((cvMatY_, cvMatY_X_3_), axis=2)
                                    cvMatsSample4 = np.concatenate((cvMatY_, cvMatY_X_4_), axis=2)
                                    samplesList   = [cvMatsSample2,cvMatsSample3,cvMatsSample4]
                                    # Sample刻写
                                    for sample in samplesList:
                                        key = ksMod % pID;
                                        key = key.encode('ascii')
                                        patchfr = Image.fromarray(np.uint8(sample)) # sample with 2 channels
                                        patchfr.save(byteBuff, format='PNG') # sample in a Y_X manner
                                        patchBytes = byteBuff.getvalue()
                                        byteBuff.truncate(0)
                                        byteBuff.seek(0)
                                        txn.put(key, patchBytes)
                                        pID += 1;

                                    if pID %100 == 0:
                                        print(str(pID)+' ',end='')
                                        innerCounter += 1
                                        if innerCounter % 10 == 0:
                                            print('')
                                    if pID % 3000 == 0:
                                        txn.commit()
                                        txn = env.begin(write=True)
                                        print('，', end='')

                txn.commit()
                byteBuff.close()
                env.close()
            print('数据集准备完毕')
        # load data
        print('Loading dataset file info')
        self.env = lmdb.open(os.path.join(conf.DATASET_DIR, conf.DATASET_FN), max_readers=3, readonly=True)
        self.txn = self.env.begin(write=False)    #  patchWanted = txn.get(b'%010d'%random.randint(1,TOTAL_NUM))
        self.len = self.txn.stat()['entries']
        print('The data prep is finished')
    def __getitem__(self, idx):
        if idx >= self.len:
            return None
        keyString = ksMod % idx;keyString = keyString.encode('ascii')
        valueBytes = self.txn.get(keyString)
        item = Image.open(io.BytesIO(valueBytes))
        ## channel split:
        y,x = item.split()
        x = self.transformer(x)
        y = self.transformer(y)
        return y,x
    def __len__(self):
        if self.txn is None:
            return -1
        else:

            return self.len



#############################################      测试代码     #############################################

# with lmdb.open(os.path.join(conf.DATASET_DIR,conf.DATASET_FN),map_size=1099511627776) as env:
#     txn = env.begin(write=True) # return the transaction
#     for i in range(TOTAL_NUM):
#         IDCounter += 1
#         value = np.ones([800*800*5],dtype=np.float32)
#         keyString = ksMod%IDCounter
#         keyString = keyString.encode('ascii')
#         txn.put(key=keyString, value=value)
#         if IDCounter % 50 == 0:
#             txn.commit()
#             print('%s is ready' % keyString)
#             txn = env.begin(write=True)
#     txn.commit()
#     env.close()
#

#
# with lmdb.open(os.path.join(conf.DATASET_DIR,conf.DATASET_FN),max_readers=2,readonly=True,map_size=1099511627776) as env:
#     st = time.time()
#     txn = env.begin(write=False)
#     print('Total entry num is:%d'%txn.stat()['entries'])
#     print('Total test num is:%d'%TEST_NUM)
#     print('The test elem size is: %d KB'%((800*800*5*4)/1024))
#     for i in range(TEST_NUM):
#         z = txn.get(b'%010d'%random.randint(1,TOTAL_NUM))
#     et = time.time()
#     dur = et - st
#     m =  TEST_NUM / dur
#     print('%d Hz'%m)
#
#
