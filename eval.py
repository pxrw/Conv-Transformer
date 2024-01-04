import torch
import torch.nn as nn
import os, argparse
import numpy as np
from tqdm import tqdm
from utils.tool import post_process_depth, flip_lr, compute_errors
from model.build_model_ori import Attn2AttnDepth
from dataset.dataloader_kitti import AttnDataLoader
from utils.opt_kitti import args


def eval(model, dataloader_eval, post_process=False):
    eval_measures = torch.zeros(10).cuda()
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda())
            gt_depth = eval_sample_batched['depth']
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth: continue

            pred_depth = model(image)
            if post_process:
                image_flipped = flip_lr(image)
                pred_depth_flipped = model(image_flipped)
                pred_depth = post_process_depth(pred_depth, pred_depth_flipped)
            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:9] += torch.tensor(measures).cuda()
        eval_measures[9] += 1

    eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures_cpu[9].item()
    eval_measures_cpu /= cnt
    print('Computing errors for {} eval samples'.format(int(cnt)), ', post_process: ', post_process)
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                 'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                 'd3'))

    for i in range(8):
        print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
    print('{:7.4f}'.format(eval_measures_cpu[8]))
    return eval_measures_cpu

def prepare_model(ckpt_path):
    model = Attn2AttnDepth(args)
    model = nn.DataParallel(model)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    print('load ckpt success...')
    return model.eval().cuda() if torch.cuda.is_available() else model.eval()


if __name__ == '__main__':
    ckpt_path = 'attn2attn3_4/PDepth/model-81374-best_abs_rel_0.06195'
    model = prepare_model(ckpt_path)
    print("== Model Initialized")
    dataloader_eval = AttnDataLoader(args, mode='online_eval')
    with torch.no_grad():
        eval_measures = eval(model, dataloader_eval, post_process=True)


