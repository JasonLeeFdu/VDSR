import os
import random
_PROJECT_BASEPATH = os.path.abspath(os.path.dirname(__file__)+os.path.sep+"..")
TRAIN_DIR = os.path.join(_PROJECT_BASEPATH,'Data/Train/')
DATASET_DIR = os.path.join(_PROJECT_BASEPATH,'Data/Dataset')
DATASET_FN = 'srdb'
SRDS291_PATH = os.path.join(_PROJECT_BASEPATH,'Data/291/')
PATCH_SIZE = 48
NUM_WORKERS = 3
MAX_Epoch = 80
MODEL_PATH = os.path.join(_PROJECT_BASEPATH,'Data/model')
SAVE_INTERVAL = 500
PRINT_INTERVAL = 100
SUM_INTERVAL =50
GPU_FLAG = True
GPUS = 0
SEED = random.randint(1, 900000)
LOG_PATH = os.path.join(_PROJECT_BASEPATH,'Log')
LR_INTERVAL = 20
WEIGHT_DECAY= 0.0#0.0000001
MOMENTUM = 0.9

## Hyper parameters concerning with training performance and Gradient Deminish or ex
## GRADIENT_CLIP = 0.1                      # small
LR = 1e-4#3                                # small 0.0005
BATCH_SIZE = 64                             # X
WEIGHT_INIT_STDDEV_FACTOR = 1.43            # big
SUMMARY_SCALAR_FIX  = 3e-3
GRADIENT_CLIP_THETA = 0.2


