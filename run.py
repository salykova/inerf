import os
import torch
import imageio
import numpy as np
import cv2
from utils import config_parser, load_blender, show_img, find_POI, img2mse, load_llff_data
from nerf_helpers import load_nerf
from render_helpers import render, to8b, get_rays
from inerf_helpers import camera_transf


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)


def run():

    # Parameters
    parser = config_parser()
    args = parser.parse_args()
    output_dir = args.output_dir
    model_name = args.model_name
    obs_img_num = args.obs_img_num
    start_pose_num = args.start_pose_num
    batch_size = args.batch_size
    kernel_size = args.kernel_size
    lrate = args.lrate
    dataset_type = args.dataset_type
    sampling_strategy = args.sampling_strategy

    # Load and pre-process the observed image
    # obs_img - rgb image with elements in range 0...255
    if dataset_type == 'blender':
        obs_img, hwf, start_pose, obs_img_pose = load_blender(args.obs_imgs_dir, obs_img_num,
                                                    start_pose_num, args.half_res, args.white_bkgd)
        H, W, focal = hwf
        near, far = 2., 6.  # Blender
    else:
        obs_img, hwf, start_pose, obs_img_pose, bds = load_llff_data(args.obs_imgs_dir, obs_img_num, start_pose_num, factor=8, recenter=True, bd_factor=.75, spherify=False)
        H, W, focal = hwf
        H, W = int(H), int(W)
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.

    start_pose = torch.Tensor(start_pose).to(device)
    if DEBUG:
        show_img("Observed image", obs_img)

    # Find points of interest
    POI = find_POI(obs_img, DEBUG)  # xy pixel coordinates of points of interest (N x 2)

    # Filter out the points that do not fit sample region
    POI_filtered = [point for point in POI if ((((point[0]-kernel_size) >= 0) and ((point[1]-kernel_size) >= 0 )) and (((point[0]+kernel_size) <= (H-1)) and ((point[1]+kernel_size) <= (W-1))))]
    POI_filtered = np.array(POI_filtered).astype(int)

    coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, W - 1, W), np.linspace(0, H - 1, H)), -1),
                        dtype=np.int)

    # create sampling masks
    masks = np.zeros((len(POI_filtered), kernel_size, kernel_size, 2), dtype=np.int)
    step = int(kernel_size / 2)
    region_size = kernel_size ** 2
    for i in range(len(masks)):
        masks[i] = coords[POI_filtered[i][0] - step: POI_filtered[i][0] + step + 1,
                   POI_filtered[i][1] - step: POI_filtered[i][1] + step + 1]
    masks = masks.reshape((masks.shape[0]*region_size), 2)

    # not_POI contains all points except of POI
    coords = coords.reshape(H * W, 2)
    not_POI = set(tuple(point) for point in coords) - set(tuple(point) for point in POI)
    not_POI = np.array([list(point) for point in not_POI]).astype(int)

    # Dilate the observed image
    I = args.dil_iter
    dil_obs_img = cv2.dilate(obs_img, np.ones((kernel_size, kernel_size), np.uint8), iterations=I)
    if DEBUG:
        show_img("Dilated image", dil_obs_img)

    dil_obs_img = (np.array(dil_obs_img) / 255.).astype(np.float32)
    obs_img = (np.array(obs_img) / 255.).astype(np.float32)

    # Load NeRF Model
    render_kwargs = load_nerf(args, device)
    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs.update(bds_dict)

    # Create pose transformation model
    cam_transf = camera_transf().to(device)
    optimizer = torch.optim.Adam(params=cam_transf.parameters(), lr=lrate, betas=(0.9, 0.999))

    testsavedir = os.path.join(output_dir, model_name)
    os.makedirs(testsavedir, exist_ok=True)

    for k in range(300):

        if sampling_strategy == 'random':
            rand_inds = np.random.choice(coords.shape[0], size=batch_size, replace=False)
            batch = coords[rand_inds]
            target_s = obs_img[batch[:, 0], batch[:, 1]]

        elif sampling_strategy == 'interest_points':
            if POI.shape[0] >= batch_size:
                rand_inds = np.random.choice(POI.shape[0], size=batch_size, replace=False)
                batch = POI[rand_inds]
            else:
                batch = np.zeros((batch_size, 2), dtype=np.int)
                batch[:POI.shape[0]] = POI
                rand_inds = np.random.choice(not_POI.shape[0], size=batch_size-POI.shape[0], replace=False)
                batch[POI.shape[0]:] = not_POI[rand_inds]
            target_s = obs_img[batch[:, 0], batch[:, 1]]

        elif sampling_strategy == 'interest_regions':
            rand_inds = np.random.choice(masks.shape[0], size=batch_size, replace=False)
            batch = masks[rand_inds]
            target_s = dil_obs_img[batch[:, 0], batch[:, 1]]
        else:
            print('Unknown sampling strategy')
            return

        target_s = torch.Tensor(target_s).to(device)
        pose = cam_transf(start_pose)

        rays_o, rays_d = get_rays(H, W, focal, pose)  # (H, W, 3), (H, W, 3)
        rays_o = rays_o[batch[:, 0], batch[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[batch[:, 0], batch[:, 1]]
        batch_rays = torch.stack([rays_o, rays_d], 0)

        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                        verbose=k < 10, retraw=True,
                                        **render_kwargs)

        optimizer.zero_grad()
        loss = img2mse(rgb, target_s)
        loss.backward()
        optimizer.step()

        new_lrate = lrate * (0.8 ** ((k + 1) / 100))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        if (k + 1) % 10 == 0:
            print(pose)
            print(loss)
            with torch.no_grad():
                rgb, disp, acc, _ = render(H, W, focal, chunk=args.chunk, c2w=pose[:3, :4], **render_kwargs)
                rgb = rgb.cpu().detach().numpy()
                rgb8 = to8b(rgb)
                filename = os.path.join(testsavedir, str(k)+'.png')
                imageio.imwrite(filename, rgb8)


def overlay():
    parser = config_parser()
    args = parser.parse_args()
    output_dir = args.output_dir
    model_name = args.model_name
    obs_img_num = args.obs_img_num

    testsavedir = os.path.join(args.output_dir, model_name)
    imgs = []
    for i in range(0, 280, 10):
        if i == 0:
            img = cv2.imread(testsavedir + '/img_' + str(i) + '.png')
        else:
            img = cv2.imread(testsavedir + '/img_' + str(i-1) + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), imgs, fps=8, quality=8)
    """
    ref = cv2.imread(testsavedir + '/ref.png')

    for i in range(0, 300, 10):
        if i == 0:
            img = cv2.imread(testsavedir + '/' + str(i) + '.png')
            dst = cv2.addWeighted(img, 0.8, ref, 0.2, 0)
            cv2.imwrite(testsavedir + '/img_' + str(i) + '.png', dst)
        else:
            img = cv2.imread(testsavedir + '/' + str(i-1) + '.png')
            dst = cv2.addWeighted(img, 0.8, ref, 0.2, 0)
            cv2.imwrite(testsavedir + '/img_' + str(i-1) + '.png', dst)
    """

DEBUG = False

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    #overlay()
    run()

