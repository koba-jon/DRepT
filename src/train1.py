import os
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils

from datasets import ImageFolder2ClassesWithPathsLoader
from weights_init import weights_init
from progress import get_date, Progress


# U-Net training
def train1(args, device, gen, dis, transform):

    # ---------------------------------
    # a1. preparation
    # ---------------------------------

    # get training dataset
    dataloader = ImageFolder2ClassesWithPathsLoader(f'datasets/{args.dataset}/{args.train1_N_dir}', f'datasets/{args.dataset}/{args.train1_A_dir}', batch_size=args.train1_batch_size, transform=transform)
    print(f'total images : Normal:{dataloader.datanum1}, Anomaly:{dataloader.datanum2}\n')
    
    # set optimizer method
    gen_optimizer = torch.optim.Adam(gen.parameters(), args.lr_gen, [args.beta1, args.beta2])
    dis_optimizer = torch.optim.Adam(dis.parameters(), args.lr_dis, [args.beta1, args.beta2])

    # set loss function
    criterion_adv = nn.BCEWithLogitsLoss(reduction='mean')
    criterion_id = nn.L1Loss(reduction='mean')

    # set tensorboard
    writer = SummaryWriter(log_dir=f'checkpoints/{args.dataset}/{args.train1_label}/tensorboard')
    f = open(f'checkpoints/{args.dataset}/{args.train1_label}/tensorboard.sh', mode='w')
    f.write('tensorboard --logdir="tensorboard"')
    f.close()

    # make directory
    os.makedirs(f'checkpoints/{args.dataset}/{args.train1_label}/models', exist_ok=True)
    os.makedirs(f'checkpoints/{args.dataset}/{args.train1_label}/optims', exist_ok=True)

    # get weights
    gen.apply(weights_init)
    dis.apply(weights_init)

    # mode
    gen.train()
    dis.train()

    # ---------------------------------
    # a2. Train model
    # ---------------------------------
    print(f'\n\n--------------------------- Train Loss ({get_date()}) ----------------------------')
    prog = Progress()
    ##################################
    for global_loop in range(1, args.train1_global_loops + 1):

        gen_loss_sum = 0.0
        gen_adv_loss_sum = 0.0
        gen_id_loss_sum = 0.0
        dis_loss_sum = 0.0
        dis_real_loss_sum = 0.0
        dis_fake_loss_sum = 0.0
        save_count = 0

        # ---------------------------------
        # b1. training per phase
        # ---------------------------------
        for local_loop in range(1, args.train1_local_loops + 1):
            
            imageNI, imageAI, _, _ = dataloader()
            mini_batch_size = imageNI.shape[0]
            imageNI = imageNI.to(device)
            imageAI = imageAI.to(device)

            # -----------------------------------------------
            # c1. Generator and Discriminator training phase
            # -----------------------------------------------

            # Generator forward
            imageNAI = torch.cat((imageNI, imageAI), dim=0)
            imageNAO = gen(imageNAI)
            imageNO, imageAO = imageNAO.split([mini_batch_size, mini_batch_size], dim=0)

            # Discriminator training
            dis_fake_out = dis(imageAO.detach())
            dis_real_out = dis(imageNI)
            dis_fake_loss = criterion_adv(dis_fake_out, torch.zeros_like(dis_fake_out))
            dis_real_loss = criterion_adv(dis_real_out, torch.ones_like(dis_real_out))
            dis_loss = dis_real_loss + dis_fake_loss
            dis_optimizer.zero_grad()
            dis_loss.backward()
            dis_optimizer.step()

            # Generator training
            dis_fake_out = dis(imageAO)
            gen_adv_loss = criterion_adv(dis_fake_out, torch.ones_like(dis_fake_out))
            gen_id_loss = criterion_id(imageNI, imageNO) * args.Lambda_id
            gen_loss = gen_adv_loss + gen_id_loss
            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            # calculate total loss
            gen_loss_sum += gen_loss.item()
            gen_adv_loss_sum += gen_adv_loss.item()
            gen_id_loss_sum += gen_id_loss.item()
            dis_loss_sum += dis_loss.item()
            dis_real_loss_sum += dis_real_loss.item()
            dis_fake_loss_sum += dis_fake_loss.item()

            # ---------------------------------
            # c3. save sample image
            # ---------------------------------
            if (local_loop % 5 == 1) or (local_loop == args.train1_local_loops):
                save_count += 1
                pair = torch.cat((imageNI, imageNO, imageAI, imageAO), dim=0)
                pair = (pair * 0.5 + 0.5) * 255.0 + 0.5
                pair = pair.type(torch.uint8)
                pair = utils.make_grid(pair, nrow=mini_batch_size)
                writer.add_image(f'Training/GlobalLoop{global_loop}', pair, save_count)
        
        
        # ---------------------------------
        # b3. record loss
        # ---------------------------------
        gen_loss_ave = gen_loss_sum / args.train1_local_loops
        gen_adv_loss_ave = gen_adv_loss_sum / args.train1_local_loops
        gen_id_loss_ave = gen_id_loss_sum / args.train1_local_loops
        dis_loss_ave = dis_loss_sum / args.train1_local_loops
        dis_real_loss_ave = dis_real_loss_sum / args.train1_local_loops
        dis_fake_loss_ave = dis_fake_loss_sum / args.train1_local_loops
        ##################################
        iterations_start = (global_loop - 1) * args.train1_local_loops + 1
        iterations_end = global_loop * args.train1_local_loops
        iterations_total = args.train1_global_loops * args.train1_local_loops
        print(f'iterations: {iterations_start}-{iterations_end}/{iterations_total},  loss_gen: {gen_loss_ave:.5f},  loss_dis: {dis_loss_ave:.5f}')

        # training tensorboard
        writer.add_scalars('Training/1:Loss(All)',
            {
                'Generator': gen_loss_ave,
                'Discriminator': dis_loss_ave,
            },
            iterations_end
        )
        ##################################
        writer.add_scalars('Training/2:Loss(Generator)',
            {
                'Total': gen_loss_ave,
                'Adversarial': gen_adv_loss_ave,
                'Identity': gen_id_loss_ave,
            },
            iterations_end
        )
        ##################################
        writer.add_scalars('Training/3:Loss(Discriminator)',
            {
                'Total': dis_loss_ave,
                'Real': dis_real_loss_ave,
                'Fake': dis_fake_loss_ave,
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

    torch.save(gen.state_dict(), f'checkpoints/{args.dataset}/{args.train1_label}/models/gen.pth')
    torch.save(dis.state_dict(), f'checkpoints/{args.dataset}/{args.train1_label}/models/dis.pth')
    torch.save(gen_optimizer.state_dict(), f'checkpoints/{args.dataset}/{args.train1_label}/optims/gen.pth')
    torch.save(dis_optimizer.state_dict(), f'checkpoints/{args.dataset}/{args.train1_label}/optims/dis.pth')
    
    writer.close()
    
    print(f'\nStage1 finished!')
    print(f'------------------------------------------------------------------------------------------\n')


