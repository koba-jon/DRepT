import os
import torch
from torch.utils.data import DataLoader
from torchvision import utils
import cv2
import numpy as np

from datasets import ImageFolderWithPaths, ImageFolderGTWithPaths


# the mixing ratio of input image and heatmap (image: 0.7)
alpha = 0.7


def test(args, device, seg, transform, transformGT):

    os.makedirs(args.test_result_dir, exist_ok=True)

    print('\n-----------------------------------------------------')
    print('(1/4) test for anomaly images')
    print('-----------------------------------------------------')
    testA(args, device, seg, transform, transformGT)
    print('-----------------------------------------------------')

    print('\n-----------------------------------------------------')
    print('(2/4) test for normal images')
    print('-----------------------------------------------------')
    testN(args, device, seg, transform)
    print('-----------------------------------------------------')

    print('\n-----------------------------------------------------')
    print('(3/4) calculate Image-level AUROC')
    print('-----------------------------------------------------')
    AUROC(args, flag='Image', anomaly_path=f'{args.test_result_dir}/anomaly_score/anomaly_image.txt', normal_path=f'{args.test_result_dir}/anomaly_score/normal_image.txt')
    print('-----------------------------------------------------')

    print('\n-----------------------------------------------------')
    print('(4/4) calculate Pixel-level AUROC')
    print('-----------------------------------------------------')
    AUROC(args, flag='Pixel', anomaly_path=f'{args.test_result_dir}/anomaly_score/anomaly_pixel.txt', normal_path=f'{args.test_result_dir}/anomaly_score/normal_pixel.txt')
    print('-----------------------------------------------------')

    print(f'\nTest finished!')

    return



def testA(args, device, seg, transform, transformGT):

    # get dataset
    dataset = ImageFolderGTWithPaths(f'datasets/{args.dataset}/{args.test_A_dir}', f'datasets/{args.dataset}/{args.test_GT_dir}', transform=transform, transformGT=transformGT)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    print('total anomaly images :',len(dataloader.dataset))

    # get weights
    seg.load_state_dict(torch.load(f'checkpoints/{args.dataset}/{args.train3_label}/models/seg.pth', map_location=device))

    # evaluation mode
    seg.eval()

    test_result_in_dir = args.test_result_dir + '/anomalyI'
    test_result_out_dir = args.test_result_dir + '/anomalyO'
    test_result_heat_dir = args.test_result_dir + '/anomalyH'
    test_result_blend_dir = args.test_result_dir + '/anomalyB'
    test_result_score_dir = args.test_result_dir + '/anomaly_score'
    os.makedirs(test_result_in_dir, exist_ok=True)
    os.makedirs(test_result_out_dir, exist_ok=True)
    os.makedirs(test_result_heat_dir, exist_ok=True)
    os.makedirs(test_result_blend_dir, exist_ok=True)
    os.makedirs(test_result_score_dir, exist_ok=True)
    f = open(f'{test_result_score_dir}/anomaly_image.txt', mode='w')
    f1 = open(f'{test_result_score_dir}/anomaly_pixel.txt', mode='w')
    f2 = open(f'{test_result_score_dir}/normal_pixel.txt', mode='w')

    for i, (image, GT, path) in enumerate(dataloader):

        image = image.to(device)
        anomaly_mask = (GT.to(device) > 0.5)
        normal_mask = anomaly_mask.bitwise_not()

        # calculate loss
        anomaly_map = seg(image)
        anomaly_map = torch.sigmoid(anomaly_map)
        anomaly_score = torch.max(anomaly_map)

        print(f'<{path[0]}> anomaly_score:{anomaly_score.item():.5f}')
        f.write(f'{anomaly_score.item()}\n')

        for x in anomaly_map.masked_select(anomaly_mask).flatten():
            f1.write(f'{x.item()}\n')
        for x in anomaly_map.masked_select(normal_mask).flatten():
            f2.write(f'{x.item()}\n')

        utils.save_image(
            image,
            f'{test_result_in_dir}/{path[0]}',
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

        utils.save_image(
            anomaly_map,
            f'{test_result_out_dir}/{path[0]}',
            nrow=1,
            normalize=True,
            value_range=(0, 1),
        )

        anomaly_map = anomaly_map[0].permute(1, 2, 0).contiguous().to('cpu').detach().numpy().copy() * 255.0
        anomaly_map = np.uint8(anomaly_map)
        heatmap = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
        cv2.imwrite(f'{test_result_heat_dir}/{path[0]}', heatmap)

        image = (image[0].expand(3, args.size, args.size).permute(1, 2, 0).contiguous().to('cpu').detach().numpy().copy() * 0.5 + 0.5) * 255.0
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        blend_image = np.uint8(alpha * image + (1.0 - alpha) * heatmap)
        cv2.imwrite(f'{test_result_blend_dir}/{path[0]}', blend_image)

    f.close()
    f1.close()
    f2.close()

    return


def testN(args, device, seg, transform):

    # get dataset
    dataset = ImageFolderWithPaths(f'datasets/{args.dataset}/{args.test_N_dir}', transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    print('total normal images :',len(dataloader.dataset))

    # get weights
    seg.load_state_dict(torch.load(f'checkpoints/{args.dataset}/{args.train3_label}/models/seg.pth', map_location=device))

    # evaluation mode
    seg.eval()

    test_result_in_dir = args.test_result_dir + '/normalI'
    test_result_out_dir = args.test_result_dir + '/normalO'
    test_result_heat_dir = args.test_result_dir + '/normalH'
    test_result_blend_dir = args.test_result_dir + '/normalB'
    test_result_score_dir = args.test_result_dir + '/anomaly_score'
    os.makedirs(test_result_in_dir, exist_ok=True)
    os.makedirs(test_result_out_dir, exist_ok=True)
    os.makedirs(test_result_heat_dir, exist_ok=True)
    os.makedirs(test_result_blend_dir, exist_ok=True)
    os.makedirs(test_result_score_dir, exist_ok=True)
    f = open(f'{test_result_score_dir}/normal_image.txt', mode='w')

    for i, (image, path) in enumerate(dataloader):

        image = image.to(device)

        anomaly_map = seg(image)
        anomaly_map = torch.sigmoid(anomaly_map)
        anomaly_score = torch.max(anomaly_map)

        print(f'<{path[0]}> anomaly_score:{anomaly_score.item():.5f}')
        f.write(f'{anomaly_score.item()}\n')

        utils.save_image(
            image,
            f'{test_result_in_dir}/{path[0]}',
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

        utils.save_image(
            anomaly_map,
            f'{test_result_out_dir}/{path[0]}',
            nrow=1,
            normalize=True,
            value_range=(0, 1),
        )

        anomaly_map = anomaly_map[0].permute(1, 2, 0).contiguous().to('cpu').detach().numpy().copy() * 255.0
        anomaly_map = np.uint8(anomaly_map)
        heatmap = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
        cv2.imwrite(f'{test_result_heat_dir}/{path[0]}', heatmap)

        image = (image[0].expand(3, args.size, args.size).permute(1, 2, 0).contiguous().to('cpu').detach().numpy().copy() * 0.5 + 0.5) * 255.0
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        blend_image = np.uint8(alpha * image + (1.0 - alpha) * heatmap)
        cv2.imwrite(f'{test_result_blend_dir}/{path[0]}', blend_image)

    f.close()

    return


def AUROC(args, flag, anomaly_path, normal_path):
        
    ANOMALY = 'Anomaly'
    NORMAL = 'Normal'

    # (1) Set Directory and Path
    result_path = f'{args.test_result_dir}/{flag}-accuracy.csv'

    # (2) Set Data
    data_all = []
    anomaly_data = 0
    normal_data = 0

    # (2.1) Set Anomaly Data
    f = open(anomaly_path, mode='r')
    while True:
        line = f.readline()
        if not line:
            break
        data = float(line)
        data_all.append([ANOMALY, data])
        anomaly_data += 1
    f.close()

    # (2.2) Set Normal Data
    f = open(normal_path, mode='r')
    while True:
        line = f.readline()
        if not line:
            break
        data = float(line)
        data_all.append([NORMAL, data])
        normal_data += 1
    f.close()

    # (3) Pre-Processing
    f = open(result_path, mode='w')
    f.write('index,TP,FP,TN,FN,TP-rate,FP-rate,TN-rate,FN-rate,precision,recall,specificity,accuracy,F1-score\n')
    total_data = len(data_all)
    print(f'total anomaly detection data ({flag}-level): {total_data}')

    # (4) Data Sort
    data_all = sorted(data_all, key=lambda x:(x[1]))

    # (5) Anomaly Detection
    AUC = 0.0
    TP = anomaly_data
    FP = normal_data
    TN = 0
    FN = 0
    index = 1
    pre_TP_rate = 1.0
    pre_FP_rate = 1.0
    for data in data_all:

        # (5.1) Update TP, FN, TN and FP
        if data[0] == ANOMALY:
            TP -= 1
            FN += 1
        else:
            FP -= 1
            TN += 1

        # (5.2) Calculation of Accuracy
        TP_rate = float(TP) / float(anomaly_data)
        FP_rate = float(FP) / float(normal_data)
        TN_rate = float(TN) / float(normal_data)
        FN_rate = float(FN) / float(anomaly_data)
        precision = float(TP) / float(TP + FP) if (TP + FP != 0) else 0.0
        recall = float(TP) / float(TP + FN)
        specificity = float(TN) / float(FP + TN)
        accuracy = float(TP + TN) / float(total_data)
        F = float(TP) / (float(TP) + 0.5 * float(FP + FN))
        AUC += (pre_TP_rate + TP_rate) * (pre_FP_rate - FP_rate) * 0.5
        pre_TP_rate = TP_rate
        pre_FP_rate = FP_rate

        # (5.3) File Output
        f.write(f'{index},')
        f.write(f'{TP},{FP},{TN},{FN},')
        f.write(f'{TP_rate},{FP_rate},{TN_rate},{FN_rate},')
        f.write(f'{precision},{recall},{specificity},')
        f.write(f'{accuracy},{F}\n')
        
        index += 1

    # (6) File Output
    f.write('\n')
    f.write(f'AUROC,{AUC}\n')
    f.close()

    f2 = open(f'{args.test_result_dir}/{flag}-AUROC.txt', mode='w')
    f2.write(f'{AUC:.3f}  {AUC:.16f}')
    f2.close()

