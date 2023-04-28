import os
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils

from datasets import ImageFolderGMMWithPathsLoader
from weights_init import weights_init
from progress import get_date, Progress


# U-Net training
def train3(args, device, f_ano, seg, transform):

    # ---------------------------------
    # a1. preparation
    # ---------------------------------

    # get training dataset
    dataloader = ImageFolderGMMWithPathsLoader(f'datasets/{args.dataset}/{args.train3_dir}', f'checkpoints/{args.dataset}/{args.train2_label}/models_as_source', batch_size=args.train3_batch_size, transform=transform)
    print(f'total images : Normal:{dataloader.datanumN}, GMM:{dataloader.datanumGMM}\n')
    
    # set optimizer method
    optimizer = torch.optim.Adam(seg.parameters(), args.lr_seg, [args.beta1, args.beta2])

    # set loss function
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    # set tensorboard
    writer = SummaryWriter(log_dir=f'checkpoints/{args.dataset}/{args.train3_label}/tensorboard')
    f = open(f'checkpoints/{args.dataset}/{args.train3_label}/tensorboard.sh', mode='w')
    f.write('tensorboard --logdir="tensorboard"')
    f.close()

    # make directory
    os.makedirs(f'checkpoints/{args.dataset}/{args.train3_label}/models', exist_ok=True)
    os.makedirs(f'checkpoints/{args.dataset}/{args.train3_label}/optims', exist_ok=True)

    # get weights
    seg.apply(weights_init)

    # mode
    f_ano.eval()
    seg.train()

    # ---------------------------------
    # a2. training model
    # ---------------------------------
    print(f'\n\n--------------------------- Train Loss ({get_date()}) ----------------------------')
    prog = Progress()
    ##################################
    for global_loop in range(1, args.train3_global_loops + 1):

        seg_loss_sum = 0.0
        save_count = 0

        # ---------------------------------
        # b1. training per phase
        # ---------------------------------
        for local_loop in range(1, args.train3_local_loops + 1):
            
            imageN, gmm_params, _, _ = dataloader(device)
            mini_batch_size = imageN.shape[0]

            # -----------------------------------------------
            # c0. Preprocessing
            # -----------------------------------------------

            # (1) Augment
            gmm_params = augmentation(gmm_params, mini_batch_size, device)

            # (2) Remove superfluous Gaussian distribution
            gmm_params = f_ano.for_target(imageN, gmm_params, thresh=args.thresh_rem)

            # (3) Build anomaly images and binary mask
            imageA, mask = f_ano(imageN, gmm_params)
            maskB = (mask > args.thresh_GT).to(torch.float)

            # -----------------------------------------------
            # c1. Discriminator training phase
            # -----------------------------------------------

            # Discriminator training
            out = seg(imageA)
            seg_loss = criterion(out, maskB)
            optimizer.zero_grad()
            seg_loss.backward()
            optimizer.step()

            # ---------------------------------
            # c2. record loss
            # ---------------------------------
            
            # calculate loss
            seg_loss_sum += seg_loss.item()

            # ---------------------------------
            # c3. save sample image
            # ---------------------------------
            if (local_loop % 5 == 1) or (local_loop == args.train3_local_loops):
                save_count += 1
                imageN = imageN * 0.5 + 0.5
                imageA = imageA * 0.5 + 0.5
                pair = torch.cat((imageN, imageA, torch.sigmoid(out).expand_as(imageN), maskB.expand_as(imageN)), dim=0)
                pair = pair * 255.0 + 0.5
                pair = pair.type(torch.uint8)
                pair = utils.make_grid(pair, nrow=mini_batch_size)
                writer.add_image(f'Training/GlobalLoop{global_loop}', pair, save_count)
        
        # ---------------------------------
        # b3. record loss
        # ---------------------------------
        seg_loss_ave = seg_loss_sum / args.train3_local_loops
        ##################################
        iterations_start = (global_loop - 1) * args.train3_local_loops + 1
        iterations_end = global_loop * args.train3_local_loops
        iterations_total = args.train3_global_loops * args.train3_local_loops
        print(f'iterations: {iterations_start}-{iterations_end}/{iterations_total},  loss: {seg_loss_ave:.5f}')

        # training tensorboard
        writer.add_scalars('Training/Loss',
            {
                'Loss': seg_loss_ave,
            },
            iterations_end
        )
        
        # ---------------------------------
        # b4. show elapsed time
        # ---------------------------------
        if global_loop % 10 == 0:
            elap, times_per_loop, rem, now_date, fin_date = prog.nab(global_loop, args.train1_global_loops)
            print(f'elapsed = {elap}({times_per_loop:.2f}sec/global_loop)   remaining = {rem}   now = {now_date}   finish = {fin_date}')
            print(f'------------------------------------------------------------------------------------------')


    torch.save(seg.state_dict(), f'checkpoints/{args.dataset}/{args.train3_label}/models/seg.pth')
    torch.save(optimizer.state_dict(), f'checkpoints/{args.dataset}/{args.train3_label}/optims/seg.pth')
    
    writer.close()

    print(f'\nStage3 finished!')
    print(f'------------------------------------------------------------------------------------------\n')



def augmentation(gmm_params, mini_batch_size, device):

    color, mu_x, mu_y, sigma_x, sigma_y, rho, pi, scale = gmm_params
    # -----------------------------------------------
    color = change_hue(color, value_range=(-1.0, 1.0))
    # -----------------------------------------------
    flip_x_mask = (torch.rand(mini_batch_size, 1, 1, 1, 1) > 0.5).to(torch.float).to(device)
    trans_x = torch.rand(mini_batch_size, 1, 1, 1, 1).to(device) * 2.0 - 1.0
    mu_x = -mu_x * flip_x_mask + mu_x * (1.0 - flip_x_mask) + trans_x
    rho = -rho * flip_x_mask + rho * (1.0 - flip_x_mask)
    # -----------------------------------------------
    flip_y_mask = (torch.rand(mini_batch_size, 1, 1, 1, 1) > 0.5).to(torch.float).to(device)
    trans_y = torch.rand(mini_batch_size, 1, 1, 1, 1).to(device) * 2.0 - 1.0
    mu_y = -mu_y * flip_y_mask + mu_y * (1.0 - flip_y_mask) + trans_y
    rho = -rho * flip_y_mask + rho * (1.0 - flip_y_mask)
    # -----------------------------------------------
    gmm_params = (color, mu_x, mu_y, sigma_x, sigma_y, rho, pi, scale)

    return gmm_params


def change_hue(color, value_range=(0.0, 1.0), eps=1e-9):

    color = (color - value_range[0]) / (value_range[1] - value_range[0])
    
    r, g, b = torch.split(color, 1, dim=1)
    max_rgb, argmax_rgb = color.max(dim=1, keepdim=True)
    min_rgb, argmin_rgb = color.min(dim=1, keepdim=True)

    sub = max_rgb - min_rgb + eps
    h1 = 60.0 * (g - r) / sub + 60.0
    h2 = 60.0 * (b - g) / sub + 180.0
    h3 = 60.0 * (r - b) / sub + 300.0

    #h = torch.cat((h2, h3, h1), dim=1).gather(dim=1, index=argmin_rgb)
    h = torch.cat((h2, h3, h1), dim=1).gather(dim=1, index=argmin_rgb) + torch.rand(color.shape[0], 1, 1, 1, 1).to(color.device) * 360.0
    s = sub / (max_rgb + eps)
    v = max_rgb

    h_ = (h - torch.floor(h / 360.0) * 360.0) / 60.0
    c = s * v
    x = c * (1.0 - torch.abs(torch.fmod(h_, 2.0) - 1.0))

    zero = torch.zeros_like(c)
    y = torch.stack((
        torch.cat((c, x, zero), dim=1),
        torch.cat((x, c, zero), dim=1),
        torch.cat((zero, c, x), dim=1),
        torch.cat((zero, x, c), dim=1),
        torch.cat((x, zero, c), dim=1),
        torch.cat((c, zero, x), dim=1),
    ), dim=0)

    idx = torch.repeat_interleave(torch.floor(h_), repeats=3, dim=1).unsqueeze(0).to(torch.long)
    idx = (idx != 6) * idx + (idx == 6) * torch.full_like(idx, 5)
    color = (y.gather(dim=0, index=idx) + (v - c)).squeeze(dim=0)

    return color * (value_range[1] - value_range[0]) + value_range[0]

