import torch
import numpy as np
from inerf_helpers import camera_transf

torch.set_default_tensor_type('torch.cuda.FloatTensor')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

transf = camera_transf().to(device)
pose = [[-0.9921147227287292, 0.09199368208646774, -0.08512099832296371, -0.34313371777534485],
                 [-0.12533323466777802, -0.7282049655914307, 0.6738020777702332, 2.7161829471588135],
                 [0.0, 0.6791574358940125, 0.7339926958084106, 2.9588191509246826],
                 [0.0, 0.0, 0.0, 1.0]]

pose = np.array(pose).astype(np.float32)
pose = (torch.Tensor(pose)).to(device)
print(transf(pose))