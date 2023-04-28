import os
from torchvision import utils
from datasets import getImageAndGMM

def transfer(args, device, f_ano, transform):

    # ---------------------------------
    # a1. preparation
    # ---------------------------------

    # make directory
    os.makedirs(args.transfer_result_dir, exist_ok=True)

    f_ano.eval()

    imageN, gmm_params = getImageAndGMM(args.transfer_image_path, args.transfer_gmm_path, device, transform)
    imageA, _ = f_ano(imageN, gmm_params)

    utils.save_image(
        imageN,
        f"{args.transfer_result_dir}/TargetNormal.png",
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )

    utils.save_image(
        imageA,
        f"{args.transfer_result_dir}/TargetAnomaly.png",
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )

    print(f'Transfer finished!')

