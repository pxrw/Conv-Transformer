import torch
import cv2, sys, os
import numpy as np
import torch.nn as nn
from path import Path
from collections import OrderedDict
import datetime
import imageio
from torchvision import transforms


def block_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__

def make_mask(depths, crop_mask, dataset):
    valid_mask = depths > 0.001
    if dataset == 'KITTI':
        if crop_mask.size(0) != valid_mask.size(0):
            crop_mask = crop_mask[0: valid_mask.size(0), ...]
        final_mask = crop_mask | valid_mask
    else:
        final_mask = valid_mask
    return valid_mask, final_mask

def scale_invariant_loss(valid_out, valid_gt):
    logdiff = torch.log(valid_out) - torch.log(valid_gt)
    return torch.sqrt((logdiff ** 2).mean() - 0.85*(logdiff.mean() ** 2)) * 10

def flip_lr(image):
    '''
    简单的左右翻转图像方法
    '''
    assert image.dim() == 4, 'you need to provide a [B, C, H, W] image to flip...'
    return torch.flip(image, [3])

def fuse_inv_depth(inv_depth, inv_depth_hat, method='mean'):
    if method == 'mean':
        return 0.5 * (inv_depth + inv_depth_hat)
    elif method == 'max':
        return torch.max(inv_depth, inv_depth_hat)
    elif method == 'min':
        return torch.min(inv_depth, inv_depth_hat)
    else:
        raise ValueError ('Unkonwn post-process method {}'.format(method))

def post_process_depth(depth, depth_flipped, method='mean'):
    B, C, H, W = depth.shape
    inv_depth_hat = flip_lr(depth_flipped)
    inv_depth_fused = fuse_inv_depth(depth, inv_depth_hat, method=method)
    xs = torch.linspace(0., 1., W, device=depth.device, dtype=depth.dtype).repeat(B, C, H, 1)
    mask = 1.0 - torch.clamp(20. * (xs - 0.05), 0., 1.)
    mask_hat = flip_lr(mask)
    return mask_hat * depth + mask * inv_depth_hat + (1.0 - mask - mask_hat) * inv_depth_fused

eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']

def compute_errors(gt, pred):
    '''
    这是pixelformer里面的error计算方法
    '''
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    # np.log10()以10为底
    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]

def normalize_result(value, vmin=None, vmax=None):
    value = value.cpu().numpy()[0, ...]
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0
    return np.expand_dims(value, 0)

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_pred, depth_gt, mask):
        d = torch.log(depth_pred[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    # [b, h//win_size, win_size, w//win_size, win_size, c] -> [b, h//win_size, w//win_size, win_size, win_size, c]
    # -> [b * h' * w', win_size, win_size, c]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows

def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x

