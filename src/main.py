import os
import random
import numpy as np
import torch
from torchvision import transforms

from networks import UNet, PatchGAN_Discriminator
from gmm import AnomalyGenerator, ConstantAnomalyGenerator
from train1 import train1
from train2 import train2
from train3 import train3
from test import test
from transfer import transfer
from options import options


def set_options(parser, mode):
    os.makedirs(f'checkpoints/{args.dataset}/{mode}', exist_ok=True)
    f = open(f'checkpoints/{args.dataset}/{mode}/options.txt', mode='w')
    f.write('----------------- Options ---------------\n')
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = f'\t[default: {str(default)}]'
        f.write(f'{str(k)}: {str(v)}{comment}\n')
    f.write('------------------- End -----------------\n\n')
    f.close()


def save_model_params(models, names, mode):
    os.makedirs(f'checkpoints/{args.dataset}/{mode}/model_params', exist_ok=True)
    for i, model in enumerate(models):
        f = open(f'checkpoints/{args.dataset}/{mode}/model_params/{names[i]}.txt', mode='w')
        num_params = 0
        for param in model.parameters():
            num_params += param.numel()
        f.write(f'Total number of parameters : {num_params/1e6:.3f} M\n\n')
        f.write(f'{model}\n')
        f.close()


if __name__ == '__main__':

    parser, args = options()

    device = torch.device(f'cuda:{args.gpu_id}' if (torch.cuda.is_available())and(args.gpu_id>=0) else 'cpu')
    print(f'using device : {device}')

    # set the seed of random number
    if args.seed_random:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # set transforms
    transform_list = [
            transforms.Resize((args.size, args.size), transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
    ]
    if args.nc == 1:
        transform_list = [transforms.Grayscale(1)] + transform_list
    transform = transforms.Compose(transform_list)


    transform_listGT = [
            transforms.Grayscale(1),
            transforms.Resize((args.size, args.size), transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
    ]
    transformGT = transforms.Compose(transform_listGT)


    if args.train1:
        gen = UNet(args, gen=True).to(device)
        dis = PatchGAN_Discriminator(args).to(device)
        set_options(parser, mode=args.train1_label)
        save_model_params([gen, dis], ['gen', 'dis'], mode=args.train1_label)
        train1(args, device, gen, dis, transform)
        
    if args.train2:
        gen = UNet(args, gen=True).to(device)
        f_ano = AnomalyGenerator(args).to(device)
        set_options(parser, mode=args.train2_label)
        save_model_params([gen, f_ano], ['gen', 'f_ano'], mode=args.train2_label)
        train2(args, device, gen, f_ano, transform)

    if args.train3:
        f_ano = ConstantAnomalyGenerator().to(device)
        seg = UNet(args, gen=False).to(device)
        set_options(parser, mode=args.train3_label)
        save_model_params([f_ano, seg], ['f_ano', 'seg'], mode=args.train3_label)
        train3(args, device, f_ano, seg, transform)
        
    if args.test:
        seg = UNet(args, gen=False).to(device)
        set_options(parser, mode="test")
        save_model_params([seg], ['seg'], mode="test")
        test(args, device, seg, transform, transformGT)
        
    if args.transfer:
        f_ano = ConstantAnomalyGenerator().to(device)
        transfer(args, device, f_ano, transform)

