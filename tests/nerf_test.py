import importlib
import os, sys
import numpy as np
import imageio
import json
import random
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
#%%

datadir = "./data/nerf_synthetic/lego"
half_res = True
testskip = 8

images, poses, render_poses, hwf, i_split = load_blender_data(datadir, half_res, testskip)
print('Loaded blender', images.shape, render_poses.shape, hwf, datadir)
#%%
#i_split = 4
#images = np.array(images[...,:3]*images[...,-1:] + (1.-images[...,-1:]), copy=True)
img = images[40]
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
