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
import utils
import tensorboardX as tbx



'''
版本性能
1.基本训练功能
2.结果保存，断点续训
----------------------------------------------
A 可视化：Tensorboard https://github.com/lanpa/tensorboardX
B 保存读取，终端显示
C 梯度修剪
'''
################
'''
注意与总结：
1.不一定非要使用PIL，而是如果想使用torchvision.transforms的相关图像变换功能的时候，需要输入是PIL。其他情况的话，其实numpy转
  torch.tensor也是很方便的。建议不是太复杂的变换，一般采用opencv+手动transform的方式。
2. PyTorch use NCHW

'''

def main():
    # load the dataset
    srDs = dm.SRDB291()
    # load the network
    net = nt.VDSR
    if not os.path.exists(conf.MODEL_PATH):
        os.mkdir(conf.MODEL_PATH)
    if not os.path.exists(conf.LOG_PATH):
        os.mkdir(conf.LOG_PATH)
    sumWriter = tbx.SummaryWriter(log_dir=conf.LOG_PATH)



    net_stat,epoch,iterr,globalStep = utils.loadLatestCheckpoint()
    if net_stat is None:
        print('No previous model found, start training now')
    else:
        net.load_state_dict(net_stat)
        print('The lastest version is Epoch %d, iter is %d，GLoablStep: %d'%(epoch,iterr,globalStep))
        #globalStep += 1


    # loss criterion
    criterion = nn.MSELoss()
    # Use GPU
    if conf.GPU_FLAG:
        if not torch.cuda.is_available():
            raise  Exception ('No GPU found or wrong gpu id. Switch the GPU_FLAG to False')
        print("Training powered by GPU, NO.{}".format(conf.GPUS))
        torch.cuda.manual_seed(conf.SEED)
        net.cuda()
        criterion.cuda()

    # get the dataset's partner ready: dataloader
    dataloader = torch.utils.data.DataLoader(srDs, batch_size=conf.BATCH_SIZE,
                                         shuffle=True, num_workers=conf.NUM_WORKERS)

    # optimizer
    if conf.WEIGHT_DECAY == 0:
        optimizer = optim.Adam([
            {'params': net.conv1.weight, 'lr': 1*conf.LR},
            {'params': net.conv1.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv2.weight, 'lr': 1*conf.LR},
            {'params': net.conv2.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv3.weight, 'lr': 1*conf.LR},
            {'params': net.conv3.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv4.weight, 'lr': 1*conf.LR},
            {'params': net.conv4.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv5.weight, 'lr': 1*conf.LR},
            {'params': net.conv5.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv6.weight, 'lr': 1*conf.LR},
            {'params': net.conv6.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv7.weight, 'lr': 1*conf.LR},
            {'params': net.conv7.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv8.weight, 'lr': 1*conf.LR},
            {'params': net.conv8.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv9.weight, 'lr': 1*conf.LR},
            {'params': net.conv9.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv10.weight, 'lr': 1*conf.LR},
            {'params': net.conv10.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv11.weight, 'lr': 1*conf.LR},
            {'params': net.conv11.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv12.weight, 'lr': 1*conf.LR},
            {'params': net.conv12.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv13.weight, 'lr': 1*conf.LR},
            {'params': net.conv13.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv14.weight, 'lr': 1*conf.LR},
            {'params': net.conv14.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv15.weight, 'lr': 1*conf.LR},
            {'params': net.conv15.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv16.weight, 'lr': 1*conf.LR},
            {'params': net.conv16.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv17.weight, 'lr': 1*conf.LR},
            {'params': net.conv17.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv18.weight, 'lr': 1*conf.LR},
            {'params': net.conv18.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv19.weight, 'lr': 1*conf.LR},
            {'params': net.conv19.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv20.weight, 'lr': 1*conf.LR},
            {'params': net.conv20.bias, 'lr': 0.1*conf.LR},
        ], lr=conf.LR)
    else:
        #optimizer = optim.Adam(net.parameters(),lr=conf.LR,weight_decay=conf.WEIGHT_DECAY)
        optimizer = optim.Adam([
            {'params': net.conv1.weight, 'lr': 1*conf.LR},
            {'params': net.conv1.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv2.weight, 'lr': 1*conf.LR},
            {'params': net.conv2.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv3.weight, 'lr': 1*conf.LR},
            {'params': net.conv3.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv4.weight, 'lr': 1*conf.LR},
            {'params': net.conv4.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv5.weight, 'lr': 1*conf.LR},
            {'params': net.conv5.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv6.weight, 'lr': 1*conf.LR},
            {'params': net.conv6.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv7.weight, 'lr': 1*conf.LR},
            {'params': net.conv7.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv8.weight, 'lr': 1*conf.LR},
            {'params': net.conv8.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv9.weight, 'lr': 1*conf.LR},
            {'params': net.conv9.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv10.weight, 'lr': 1*conf.LR},
            {'params': net.conv10.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv11.weight, 'lr': 1*conf.LR},
            {'params': net.conv11.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv12.weight, 'lr': 1*conf.LR},
            {'params': net.conv12.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv13.weight, 'lr': 1*conf.LR},
            {'params': net.conv13.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv14.weight, 'lr': 1*conf.LR},
            {'params': net.conv14.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv15.weight, 'lr': 1*conf.LR},
            {'params': net.conv15.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv16.weight, 'lr': 1*conf.LR},
            {'params': net.conv16.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv17.weight, 'lr': 1*conf.LR},
            {'params': net.conv17.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv18.weight, 'lr': 1*conf.LR},
            {'params': net.conv18.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv19.weight, 'lr': 1*conf.LR},
            {'params': net.conv19.bias, 'lr': 0.1*conf.LR},
            {'params': net.conv20.weight, 'lr': 1*conf.LR},
            {'params': net.conv20.bias, 'lr': 0.1*conf.LR},
        ], lr=conf.LR, weight_decay=conf.WEIGHT_DECAY)
    net.train()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=math.sqrt(0.1))
    # numerial statistic visible for training console
    AvgFreq = 0; Avgloss = 0;

    ## TRAINING EPOCH
    for epoch_pos in range(epoch):
        scheduler.step()
    for epoch_pos in range(epoch,conf.MAX_Epoch):
        print('----------------------- Epoch %d ----------------------------'%(epoch_pos))
        scheduler.step()
        for iterNum,(y,x) in enumerate(dataloader,iterr+1):
            globalStep += 1
            if conf.GPU_FLAG:
                x = x.cuda()
                y = y.cuda()

            startTime = time.time()
            #####
            optimizer.zero_grad()      #
            pred=net(x)
            loss = criterion(pred,y-x) #
            loss.backward()            #
            utils.gradientClip(net.parameters(), conf.GRADIENT_CLIP_THETA/utils.get_lr(optimizer))
            optimizer.step()
            ####


            endTime = time.time()
            lossData = loss.cpu().detach().numpy()
            loss.cuda()
            AvgFreq += endTime - startTime
            Avgloss += lossData
            if loss < 0.01:
                sumWriter.add_scalar('Data/loss', loss, globalStep)
            else:
                sumWriter.add_scalar('Data/loss', torch.Tensor(np.array(conf.SUMMARY_SCALAR_FIX)), globalStep)

            if iterNum % conf.PRINT_INTERVAL == 0 and iterNum is not 0:
                AvgFreq = (conf.PRINT_INTERVAL * conf.BATCH_SIZE) / AvgFreq
                Avgloss = Avgloss / conf.PRINT_INTERVAL
                format_str = '%s: Iters %d, average loss(255) = %.7f, GI:%d ,lr:%.7f, g:%.6f, average frequency = %.3f(HZ) (batch per sec)'
                ## get the gradient:
                g = 0;c = 0
                for param in net.parameters():
                    g += math.fabs(param.grad.data.sum())
                    c += 1.0
                g = g/c
                if iterNum % conf.SAVE_INTERVAL == 0:
                    print(format_str % (datetime.now(), iterNum, math.sqrt(Avgloss) * 255, globalStep,utils.get_lr(optimizer),g,AvgFreq), end='')
                else:
                    print(format_str % (datetime.now(), iterNum, math.sqrt(Avgloss) * 255, globalStep, utils.get_lr(optimizer),g,AvgFreq))
                AvgFreq = 0
                Avgloss = 0

            if iterNum % conf.SUM_INTERVAL == 0 and iterNum is not 0:
                sumWriter.add_image('Data/input', x, globalStep)
                sumWriter.add_image('Data/Label', y, globalStep)
                sumWriter.add_image('Data/result', pred + x, globalStep)
                sumWriter.add_image('Data/pred', pred, globalStep)
                sumWriter.add_image('Data/residualToLearn', y- x, globalStep)
                # for better visualization
                nc = np.random.randint(0,conf.BATCH_SIZE-1)
                xnc = x[nc,:,:,:]
                ync = y[nc,:,:,:]
                prnc = pred[nc,:,:,:]

                sumWriter.add_image('Vis/input', xnc, globalStep)
                sumWriter.add_image('Vis/Label', ync, globalStep)
                sumWriter.add_image('Vis/result', prnc + xnc, globalStep)
                sumWriter.add_image('Vis/pred', prnc, globalStep)
                sumWriter.add_image('Vis/residualToLearn', ync - xnc, globalStep)

            if iterNum % conf.SAVE_INTERVAL == 0 and iterNum is not 0:
                utils.saveCheckpoint(net.state_dict(),epoch_pos,iterNum,globalStep)
                print('...... SAVED')
        iterr = -1




if __name__ == '__main__':
        main()



'''




            ########## check for gradient
            xx = 0
            for param in net.parameters():
                xx += param.grad.data.sum()

            print('Gradient sum is: ' + str(xx))
            ########## check for gradient END



'''





'''
# 保存和加载整个模型
torch.save(model_object, 'model.pkl')
model = torch.load('model.pkl')
1
2
3
# 仅保存和加载模型参数(推荐使用)
torch.save(model_object.state_dict(), 'params.pkl')
model_object.load_state_dict(torch.load('params.pkl'))



### check the gradient
for param in net.parameters():
    print(param.grad.data.sum())
    

'''

#nn.utils.clip_grad_norm(net.parameters(), conf.GRADIENT_CLIP)
