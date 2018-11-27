import torch.optim
import torch.utils.data
import torch.utils.data.distributed
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
import io
import matplotlib.pyplot as plt



img = Image.open('/home/winston/workSpace/PycharmProjects/VDSR_rebuild/Data/291/000t34.bmp')
w,h = img.size
## To bytes
byteBuff = io.BytesIO()
img.save(byteBuff,format='PNG')
byte = byteBuff.getvalue()
## from bytes
im2 = Image.open(io.BytesIO(byte))
img2 = np.array(im2)

