import torch
import numpy as np
from inerf_helpers import camera_transf
import os
import json

with open(os.path.join("../data/nerf_synthetic/lego/obs_imgs", 'transforms.json'), 'r') as fp:
    meta = json.load(fp)
print(os.path.join("../data/nerf_synthetic/lego/obs_imgs", meta['frames'][0]['file_path'] + '.png'))