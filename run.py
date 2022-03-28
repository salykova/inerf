import os
import torch
import imageio
import numpy as np
import skimage
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
    batch_size = args.batch_size
    spherify = args.spherify
    kernel_size = args.kernel_size
    lrate = args.lrate
    dataset_type = args.dataset_type
    sampling_strategy = args.sampling_strategy
    delta_phi, delta_theta, delta_psi, delta_t = args.delta_phi, args.delta_theta, args.delta_psi, args.delta_t
    noise, sigma, amount = args.noise, args.sigma, args.amount
    delta_brightness = args.delta_brightness

    # Load and pre-process an observed image
    # obs_img -> rgb image with elements in range 0...255
    if dataset_type == 'blender':
        obs_img, hwf, start_pose, obs_img_pose = load_blender(args.data_dir, model_name, obs_img_num,
                                                args.half_res, args.white_bkgd, delta_phi, delta_theta, delta_psi, delta_t)
        H, W, focal = hwf
        near, far = 2., 6.  # Blender
    else:
        obs_img, hwf, start_pose, obs_img_pose, bds = load_llff_data(args.data_dir, model_name, obs_img_num, delta_phi,
                                                delta_theta, delta_psi, delta_t, factor=8, recenter=True, bd_factor=.75, spherify=spherify)
        H, W, focal = hwf
        H, W = int(H), int(W)
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.

    obs_img = (np.array(obs_img) / 255.).astype(np.float32)

    # change brightness of the observed image (to test robustness of inerf)
    if delta_brightness != 0:
        obs_img = (np.array(obs_img) / 255.).astype(np.float32)
        obs_img = cv2.cvtColor(obs_img, cv2.COLOR_RGB2HSV)
        if delta_brightness < 0:
            obs_img[..., 2][obs_img[..., 2] < abs(delta_brightness)] = 0.
            obs_img[..., 2][obs_img[..., 2] >= abs(delta_brightness)] += delta_brightness
        else:
            lim = 1. - delta_brightness
            obs_img[..., 2][obs_img[..., 2] > lim] = 1.
            obs_img[..., 2][obs_img[..., 2] <= lim] += delta_brightness
        obs_img = cv2.cvtColor(obs_img, cv2.COLOR_HSV2RGB)
        show_img("Observed image", obs_img)

    # apply noise to the observed image (to test robustness of inerf)
    if noise == 'gaussian':
        obs_img_noised = skimage.util.random_noise(obs_img, mode='gaussian', var=sigma**2)
    elif noise == 's_and_p':
        obs_img_noised = skimage.util.random_noise(obs_img, mode='s&p', amount=amount)
    elif noise == 'pepper':
        obs_img_noised = skimage.util.random_noise(obs_img, mode='pepper', amount=amount)
    elif noise == 'salt':
        obs_img_noised = skimage.util.random_noise(obs_img, mode='salt', amount=amount)
    elif noise == 'poisson':
        obs_img_noised = skimage.util.random_noise(obs_img, mode='poisson')
    else:
        obs_img_noised = obs_img

    obs_img_noised = (np.array(obs_img_noised) * 255).astype(np.uint8)
    if DEBUG:
        show_img("Observed image", obs_img_noised)

    # find points of interest of the observed image
    POI = find_POI(obs_img_noised, DEBUG)  # xy pixel coordinates of points of interest (N x 2)
    obs_img_noised = (np.array(obs_img_noised) / 255.).astype(np.float32)

    # create meshgrid from the observed image
    coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, W - 1, W), np.linspace(0, H - 1, H)), -1),
                        dtype=int)

    # create sampling mask for interest region sampling strategy
    interest_regions = np.zeros((H, W, ), dtype=np.uint8)
    interest_regions[POI[:,1], POI[:,0]] = 1
    I = args.dil_iter
    interest_regions = cv2.dilate(interest_regions, np.ones((kernel_size, kernel_size), np.uint8), iterations=I)
    interest_regions = np.array(interest_regions, dtype=bool)
    interest_regions = coords[interest_regions]

    # not_POI -> contains all points except of POI
    coords = coords.reshape(H * W, 2)
    not_POI = set(tuple(point) for point in coords) - set(tuple(point) for point in POI)
    not_POI = np.array([list(point) for point in not_POI]).astype(int)

    # Load NeRF Model
    render_kwargs = load_nerf(args, device)
    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs.update(bds_dict)

    # Create pose transformation model
    start_pose = torch.Tensor(start_pose).to(device)
    cam_transf = camera_transf().to(device)
    optimizer = torch.optim.Adam(params=cam_transf.parameters(), lr=lrate, betas=(0.9, 0.999))

    # calculate angles and translation of the observed image's pose
    phi_ref = np.arctan2(obs_img_pose[1,0], obs_img_pose[0,0])*180/np.pi
    theta_ref = np.arctan2(-obs_img_pose[2, 0], np.sqrt(obs_img_pose[2, 1]**2 + obs_img_pose[2, 2]**2))*180/np.pi
    psi_ref = np.arctan2(obs_img_pose[2, 1], obs_img_pose[2, 2])*180/np.pi
    translation_ref = np.sqrt(obs_img_pose[0,3]**2 + obs_img_pose[1,3]**2 + obs_img_pose[2,3]**2)
    #translation_ref = obs_img_pose[2, 3]

    testsavedir = os.path.join(output_dir, model_name)
    os.makedirs(testsavedir, exist_ok=True)

    # imgs - array with images are used to create a video of optimization process
    if OVERLAY is True:
        imgs = []

    for k in range(300):

        if sampling_strategy == 'random':
            rand_inds = np.random.choice(coords.shape[0], size=batch_size, replace=False)
            batch = coords[rand_inds]

        elif sampling_strategy == 'interest_points':
            if POI.shape[0] >= batch_size:
                rand_inds = np.random.choice(POI.shape[0], size=batch_size, replace=False)
                batch = POI[rand_inds]
            else:
                batch = np.zeros((batch_size, 2), dtype=np.int)
                batch[:POI.shape[0]] = POI
                rand_inds = np.random.choice(not_POI.shape[0], size=batch_size-POI.shape[0], replace=False)
                batch[POI.shape[0]:] = not_POI[rand_inds]

        elif sampling_strategy == 'interest_regions':
            rand_inds = np.random.choice(interest_regions.shape[0], size=batch_size, replace=False)
            batch = interest_regions[rand_inds]

        else:
            print('Unknown sampling strategy')
            return

        target_s = obs_img_noised[batch[:, 1], batch[:, 0]]
        target_s = torch.Tensor(target_s).to(device)
        pose = cam_transf(start_pose)

        rays_o, rays_d = get_rays(H, W, focal, pose)  # (H, W, 3), (H, W, 3)
        rays_o = rays_o[batch[:, 1], batch[:, 0]]  # (N_rand, 3)
        rays_d = rays_d[batch[:, 1], batch[:, 0]]
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

        if (k + 1) % 20 == 0 or k == 0:
            print('Step: ', k)
            print('Loss: ', loss)

            with torch.no_grad():
                pose_dummy = pose.cpu().detach().numpy()
                # calculate angles and translation of the optimized pose
                phi = np.arctan2(pose_dummy[1, 0], pose_dummy[0, 0]) * 180 / np.pi
                theta = np.arctan2(-pose_dummy[2, 0], np.sqrt(pose_dummy[2, 1] ** 2 + pose_dummy[2, 2] ** 2)) * 180 / np.pi
                psi = np.arctan2(pose_dummy[2, 1], pose_dummy[2, 2]) * 180 / np.pi
                translation = np.sqrt(pose_dummy[0,3]**2 + pose_dummy[1,3]**2 + pose_dummy[2,3]**2)
                #translation = pose_dummy[2, 3]
                # calculate error between optimized and observed pose
                phi_error = abs(phi_ref - phi) if abs(phi_ref - phi)<300 else abs(abs(phi_ref - phi)-360)
                theta_error = abs(theta_ref - theta) if abs(theta_ref - theta)<300 else abs(abs(theta_ref - theta)-360)
                psi_error = abs(psi_ref - psi) if abs(psi_ref - psi)<300 else abs(abs(psi_ref - psi)-360)
                rot_error = phi_error + theta_error + psi_error
                translation_error = abs(translation_ref - translation)
                print('Rotation error: ', rot_error)
                print('Translation error: ', translation_error)
                print('-----------------------------------')

            if OVERLAY is True:
                with torch.no_grad():
                    rgb, disp, acc, _ = render(H, W, focal, chunk=args.chunk, c2w=pose[:3, :4], **render_kwargs)
                    rgb = rgb.cpu().detach().numpy()
                    rgb8 = to8b(rgb)
                    ref = to8b(obs_img)
                    filename = os.path.join(testsavedir, str(k)+'.png')
                    dst = cv2.addWeighted(rgb8, 0.7, ref, 0.3, 0)
                    imageio.imwrite(filename, dst)
                    imgs.append(dst)

    if OVERLAY is True:
        imageio.mimwrite(os.path.join(testsavedir, 'video.gif'), imgs, fps=8) #quality = 8 for mp4 format

DEBUG = False
OVERLAY = False

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    run()
