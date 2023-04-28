import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm import tqdm

from datasets import ImageFolderWithPaths


def train2(args, device, gen, f_ano, transform):

    # get dataset
    dataset = ImageFolderWithPaths(f'datasets/{args.dataset}/{args.train2_dir}', transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    print(f'total images : {len(dataloader.dataset)}')

    # get weights
    gen.load_state_dict(torch.load(f'checkpoints/{args.dataset}/{args.train1_label}/models/gen.pth', map_location=device))

    # optimizer
    optimizer = torch.optim.Adam(f_ano.parameters(), args.lr_gmm, [args.beta1, args.beta2])

    # set loss function
    criterion = nn.MSELoss(reduction='mean')

    # evaluation mode
    gen.eval()
    f_ano.train()

    os.makedirs(f'checkpoints/{args.dataset}/{args.train2_label}/models', exist_ok=True)
    os.makedirs(f'checkpoints/{args.dataset}/{args.train2_label}/samples', exist_ok=True)

    for i, (imageRA, path) in enumerate(dataloader):

        imageRA = imageRA.to(device)
        f_ano.reset()

        # calculate loss
        imageFN = gen(imageRA).detach()
        imageS = torch.abs(imageFN - imageRA)
        imageS = (1.0 - imageS / imageS.max())
        iterator = tqdm(range(args.train2_iters))
        for j in iterator:
            imageFA, gmm_prob = f_ano(imageFN)
            rec_loss = criterion(imageFA, imageRA)
            penalty = criterion(imageS.detach() * gmm_prob, torch.zeros_like(imageS))
            loss = rec_loss + penalty * args.Lambda_fault
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iterator.set_description(
                (
                    f'data:{i + 1}/{len(dataloader.dataset)} '
                    f'loss:{loss.item():.5f} '
                )
            )
        
        fname = f'checkpoints/{args.dataset}/{args.train2_label}/models/{path[0]}'
        idx = fname.rfind(".")
        fname = fname[0:idx] + ".pth"
        torch.save(f_ano.gmm_params(), fname)

        base_path, ext = os.path.splitext(path[0])

        utils.save_image(
            imageRA,
            f'checkpoints/{args.dataset}/{args.train2_label}/samples/{base_path}_Input{ext}',
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

        utils.save_image(
            imageFN,
            f'checkpoints/{args.dataset}/{args.train2_label}/samples/{base_path}_Output{ext}',
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

        utils.save_image(
            imageFA,
            f'checkpoints/{args.dataset}/{args.train2_label}/samples/{base_path}_Output+GMM{ext}',
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

        imageGMM = f_ano.component(imageRA.shape[0], imageRA.shape[3], imageRA.shape[2], imageRA.device)[0] * 0.5 + 0.5
        imageGMM = torchvision.transforms.functional.to_pil_image(imageGMM)
        imageGMM.save(f'checkpoints/{args.dataset}/{args.train2_label}/samples/{base_path}_GMM{ext}')


    print(f'\nStage2 finished!')
    print(f'------------------------------------------------------------------------------------------\n')


