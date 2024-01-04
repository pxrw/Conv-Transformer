import os, sys, errno
import cv2
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from utils.opt import args
from utils.tool import flip_lr, post_process_depth
from model.build_model_ori import Attn2AttnDepth

if args.dataset == 'kitti':
    from dataset.dataloader_kitti import AttnDataLoader
else:
    from dataset.dataloader_nyu import AttnDataLoader


def test(args):
    args.mode = 'test'
    dataset_args = args.kitti if args.dataset == 'kitti' else args.nyu
    dataloader = AttnDataLoader(args, mode='test')
    model = Attn2AttnDepth(args)
    model = nn.DataParallel(model)
    ckpt = torch.load('kitti_without_mosaic/PDepth/model-131835-best_d1_0.96105')
    model.load_state_dict(ckpt['model'])
    print('load ckpt success...')
    model.eval().cuda()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    with open(dataset_args.filenames_file_test, 'r') as f:
        lines = f.readlines()

    num_test = len(lines)
    print('now testing {}'.format(num_test))

    pred_depths = []
    with torch.no_grad():
        for _, sample in enumerate(tqdm(dataloader.data)):
            image = torch.autograd.Variable(sample['image'].cuda())
            pred = model(image)
            post_process = True
            if post_process:
                image_flipped = flip_lr(image)
                pred_flipped = model(image_flipped)
                pred = post_process_depth(pred, pred_flipped)
            pred_depth = pred.cpu().numpy().squeeze()

            if dataset_args.do_kb_crop:
                height, width = 352, 1216
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
                pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
                pred_depth = pred_depth_uncropped
            pred_depths.append(pred_depth)

    print('predict done...')

    save_name = 'results'
    if not os.path.exists(save_name):
        try:
            os.mkdir(save_name)
            os.mkdir(save_name + '/raw')
            os.mkdir(save_name + '/cmap')
            os.mkdir(save_name + '/rgb')
            os.mkdir(save_name + '/gt')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    for s in tqdm(range(num_test)):
        filename_pred_png = save_name + '/raw/' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
        filename_cmap_png = save_name + '/cmap/' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
        filename_image_png = save_name + '/rgb/' + lines[s].split()[0].split('/')[-1]

        rgb_path = os.path.join(dataset_args.data_path_eval, lines[s].split()[0])
        image = cv2.imread(rgb_path)
        pred_depth = pred_depths[s]
        pred_depth_scaled = pred_depth * 256.0
        pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
        cv2.imwrite(filename_pred_png, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        cv2.imwrite(filename_image_png, image[10:-1 - 9, 10:-1 - 9, :])
        plt.imsave(filename_cmap_png, np.log10(pred_depth), cmap='jet')

    return


if __name__ == '__main__':
    test(args)
    # with open(args.filenames_file_test, 'r') as f:
    #     lines = f.readlines()
    # data_drive = lines[0]
    # print(data_drive)
    # print(data_drive.split()[0].split('/')[-1])
