import os
import torch
import imageio
import numpy as np
import cv2
from utils import config_parser, load_img, show_img, find_ipoints, img2mse
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

    # Load and pre-process the observed image
    # obs_img - rgb image with elements in range 0...255
    obs_img, hwf, start_pose, end_pose = load_img(args.obs_imgs_dir, obs_img_num,
                                                  start_pose_num, args.half_res, args.white_bkgd)
    start_pose = torch.Tensor(start_pose).to(device)

    if DEBUG:
        show_img("Observed image", obs_img)

    # Find points of interest
    interest_pts_coords = find_ipoints(obs_img, DEBUG)  # xy pixel coordinates of interest points (N x 2)

    # Dilate the observed image
    I = args.dil_iter
    dil_obs_img = cv2.dilate(obs_img, np.ones((kernel_size, kernel_size), np.uint8), iterations=I)
    if DEBUG:
        show_img("Dilated image", dil_obs_img)
        print(start_pose)
        print(end_pose)
        return
    dil_obs_img = (np.array(dil_obs_img) / 255.).astype(np.float32)

    H, W, focal = hwf
    region_size = kernel_size ** 2
    num_regions = int(batch_size / region_size)

    # Load NeRF Model
    near, far = 2., 6.  # Blender
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

    for k in range(600):

        batch = np.zeros((num_regions, kernel_size, kernel_size, 2), dtype=np.int)
        rand_inds = np.random.choice(interest_pts_coords.shape[0], size=[num_regions], replace=False)  # (N_rand,)
        rand_coords = interest_pts_coords[rand_inds]
        step = int(kernel_size / 2)
        coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, H - 1, H), np.linspace(0, W - 1, W)), -1),
                            dtype=np.int)  # (H, W, 2)
        for i in range(len(rand_coords)):
            if ((rand_coords[i][0] - step) < 0) or ((rand_coords[i][0] + step + 1) > H) or (
                    (rand_coords[i][1] - step) < 0) or ((rand_coords[i][1] + step + 1) > W):
                continue
            batch[i] = coords[rand_coords[i][0] - step: rand_coords[i][0] + step + 1,
                       rand_coords[i][1] - step: rand_coords[i][1] + step + 1]
        batch = batch.reshape((batch.shape[0] * region_size, 2))  # (num_regions * region_size, 2)
        target_s = dil_obs_img[batch[:, 0], batch[:, 1]]
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

        if (k + 1) % 30 == 0:
            print(pose)
            print(loss)


def run2():
    parser = config_parser()
    args = parser.parse_args()
    model_name = args.model_name

    near, far = 2., 6.
    render_kwargs = load_nerf(args, device)
    bds_dict = {
        'near': near,
        'far': far,
    }
    pose = torch.Tensor([
                [
                    -0.9997775554656982,
                    0.016899261623620987,
                    -0.012622464448213577,
                    -0.05088278651237488
                ],
                [
                    -0.02109292894601822,
                    -0.801003098487854,
                    0.5982884764671326,
                    2.411778211593628
                ],
                [
                    9.313225746154785e-10,
                    0.5984216928482056,
                    0.8011811971664429,
                    3.2296650409698486
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ])
    render_kwargs.update(bds_dict)
    H, W = 400, 400
    camera_angle_x = 0.6911112070083618
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    print('RENDER ONLY')
    rgb, disp, acc, _ = render(H, W, focal, chunk=args.chunk, c2w=pose[:3, :4], **render_kwargs)
    testsavedir = os.path.join(args.output_dir, model_name)
    os.makedirs(testsavedir, exist_ok=True)
    rgb = rgb.cpu().numpy()
    rgb8 = to8b(rgb)
    filename = os.path.join(testsavedir, '0.png')
    imageio.imwrite(filename, rgb8)

DEBUG = False

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    run()