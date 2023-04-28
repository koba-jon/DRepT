import argparse


def options():

    parser = argparse.ArgumentParser()

    # (1) Define for general parameter
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--size', type=int, default=256, help='image width and height (x>=32)')
    parser.add_argument('--nc', type=int, default=3, help='input image channel : RGB=3, grayscale=1')
    parser.add_argument('--gpu_id', type=int, default=0, help='cuda device : "x=-1" is cpu device')
    parser.add_argument('--seed_random', action='store_true', help='whether to make the seed of random number in a random')
    parser.add_argument('--seed', type=int, default=0, help='the seed of random number')

    # (2.1) Define for training - stage 1
    parser.add_argument('--train1', action='store_true', help='training mode on/off')
    parser.add_argument('--train1_N_dir', type=str, default='trainN', help='training source-normal image directory : ./datasets/<dataset>/<train1_N_dir>/<image files>')
    parser.add_argument('--train1_A_dir', type=str, default='trainA', help='training source-anomaly image directory : ./datasets/<dataset>/<train1_A_dir>/<image files>')
    parser.add_argument('--train1_global_loops', type=int, default=100, help='the number of global loop for training stage 1 : iterations = <train1_global_loops> * <train1_local_loops>')
    parser.add_argument('--train1_local_loops', type=int, default=100, help='the number of local loop for training stage 1 : iterations = <train1_global_loops> * <train1_local_loops>')
    parser.add_argument('--train1_batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--train1_label', type=str, default="stage1", help='stage label for training stage 1')

    # (2.2) Define for training - stage 2
    parser.add_argument('--train2', action='store_true', help='training mode 2 on/off')
    parser.add_argument('--train2_dir', type=str, default='trainA', help='training source-anomaly image directory : ./datasets/<dataset>/<train2_dir>/<image files>')
    parser.add_argument('--train2_iters', type=int, default=10000, help='training total iterations for each phase')
    parser.add_argument('--train2_label', type=str, default="stage2", help='stage label for training stage 2')

    # (2.3) Define for training - stage 3
    parser.add_argument('--train3', action='store_true', help='training mode 3 on/off')
    parser.add_argument('--train3_dir', type=str, default='train', help='training target-normal image directory : ./datasets/<dataset>/<train3_dir>/<image files>')
    parser.add_argument('--train3_global_loops', type=int, default=100, help='the number of global loop for training stage 3 : iterations = <train3_global_loops> * <train3_local_loops>')
    parser.add_argument('--train3_local_loops', type=int, default=100, help='the number of local loop for training stage 3 : iterations = <train3_global_loops> * <train3_local_loops>')
    parser.add_argument('--train3_batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--train3_label', type=str, default="stage3", help='stage label for training stage 3')

    # (3) Define for test
    parser.add_argument('--test', action='store_true', help='test mode on/off')
    parser.add_argument('--test_N_dir', type=str, default='testN', help='test normal image directory : ./datasets/<dataset>/<test_N_dir>/<image files>')
    parser.add_argument('--test_A_dir', type=str, default='testA', help='test anomaly image directory : ./datasets/<dataset>/<test_A_dir>/<image files>')
    parser.add_argument('--test_GT_dir', type=str, default='testGT', help='test GT image directory : ./datasets/<dataset>/<test_GT_dir>/<image files>')
    parser.add_argument('--test_result_dir', type=str, default='test_result', help='test result directory : ./<test_result_dir>')
    parser.add_argument('--test_label', type=str, default="test", help='stage label for test')

    # (4) Define for transfer
    parser.add_argument('--transfer', action='store_true', help='test mode on/off')
    parser.add_argument('--transfer_image_path', type=str, help='trasnfer image path : ./<transfer_image_path>')
    parser.add_argument('--transfer_gmm_path', type=str, help='trasnfer gmm path : ./<transfer_gmm_path>')
    parser.add_argument('--transfer_result_dir', type=str, default='transfer_result', help='trasnfer result directory : ./<transfer_result_dir>')

    # (5) Define for hyperparameter
    parser.add_argument('--K', type=int, default=30, help='the number of normal distribution in single GMM')
    parser.add_argument('--Lambda_id', type=float, default=40.0, help='importance of loss for identity function')
    parser.add_argument('--Lambda_fault', type=float, default=0.2, help='importance of loss with fault mask')
    parser.add_argument('--thresh_GT', type=float, default=0.15, help='threshold for ground truth')
    parser.add_argument('--thresh_rem', type=float, default=0.15, help='threshold for removing superfluous gaussian distribution')
    parser.add_argument('--lr_gen', type=float, default=2e-4, help='learning rate for generator')
    parser.add_argument('--lr_dis', type=float, default=2e-4, help='learning rate for discriminator')
    parser.add_argument('--lr_gmm', type=float, default=5e-3, help='learning rate for GMM parameter')
    parser.add_argument('--lr_seg', type=float, default=2e-4, help='learning rate for segmentation network')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta 1 in Adam of optimizer method')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta 2 in Adam of optimizer method')
    parser.add_argument('--nf', type=int, default=64, help='the number of filters in convolution layer closest to image')

    args = parser.parse_args()

    return parser, args
