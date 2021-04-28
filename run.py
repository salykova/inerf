import os
import torch
import imageio
import numpy as np
import cv2
from utils import config_parser, load_img, show_img, find_ipoints
from nerf_helpers import get_embedder, NeRF, run_network
from render_helpers import render, to8b

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    # Disable updating of weights
    for param in model.parameters():
        param.requires_grad = False
    for param in model_fine.parameters():
        param.requires_grad = False
    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

def run():
    # Parameters
    parser = config_parser()
    args = parser.parse_args()
    basedir = args.basedir
    expname = args.expname

    # Loading and processing of observed image
    obs_img_numb = 1
    obs_img, hwf = load_img(os.path.join(args.oidir, 'obs_img_' + str(obs_img_numb) + '.png'),
                                  args.half_res, args.white_bkgd)  # rgb image with elements in range 0...255
    if DEBUG:
        show_img("Observed image", obs_img)

    ipoints_coords = find_ipoints(obs_img, DEBUG) # xy pixel coordinates of interest points (N x 2)
    I = 2
    kernel_size = 3

    dil_obs_img = cv2.dilate(obs_img, np.ones((kernel_size, kernel_size), np.uint8), iterations = I)
    if DEBUG:
        show_img("Dilated image", dil_obs_img)

    dil_obs_img = (np.array(dil_obs_img) / 255.).astype(np.float32)

    # Load NeRF Model
    near, far = 2., 6.
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start
    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    #end_pose = [[-0.999970555305481, 0.0056571876630187035, -0.00517683569341898, -0.020868491381406784],
                #[-0.007668337319046259,-0.7377116084098816, 0.675072431564331, 2.721303939819336],
                #[-4.6566125955216364e-10, 0.675092339515686, 0.7377332448959351, 2.973897933959961],
                #[0.0, 0.0, 0.0, 1.0]]
    #end_pose = np.array(end_pose).astype(np.float32)
    #end_pose = torch.Tensor(end_pose).to(device)

    start_pose = [[-0.999970555305481, 0.0056571876630187035, -0.00517683569341898, -0.020868491381406784],
                [-0.007668337319046259,-0.7377116084098816, 0.675072431564331, 2.721303939819336],
                [-4.6566125955216364e-10, 0.675092339515686, 0.7377332448959351, 2.973897933959961],
                [0.0, 0.0, 0.0, 1.0]]
    start_pose = np.array(start_pose).astype(np.float32)
    start_pose = torch.Tensor(start_pose).to(device)

    H, W, focal = hwf
    H, W = int(H), int(W)

    batch_size = 2048
    region_size = kernel_size ** 2
    region_num = int(batch_size / region_size)


    print('RENDER ONLY')
    rgb, disp, acc, _ = render(H, W, focal, chunk=args.chunk, c2w=start_pose[:3,:4], **render_kwargs_test)
    testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
    os.makedirs(testsavedir, exist_ok=True)
    rgb = rgb.cpu().numpy()
    rgb8 = to8b(rgb)
    filename = os.path.join(testsavedir, '0.png')
    imageio.imwrite(filename, rgb8)

DEBUG = False

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    run()