import torch, cv2, os
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from matplotlib import pyplot as plt
from utils.opt import args
from utils.tool import flip_lr, post_process_depth
from dataset.dataloader_kitti import AttnDataLoader
from model.build_model_ori import Attn2AttnDepth

if args.dataset == 'kitti':
    dataset_args = args.kitti
else:
    dataset_args = args.nyu

def predict_one_image(model, image, post_process=True):
    with torch.no_grad():
        image_pred = model(image)
        if post_process:
            image_flipped = flip_lr(image)
            pred_flipped = model(image_flipped)
            image_pred = post_process_depth(image_pred, pred_flipped)
        pred_depth = image_pred.cpu().numpy().squeeze()

        if dataset_args.do_kb_crop is True:
            height, width = 352, 1216
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped
    return pred_depth * 256.0

def process_image(image_path):
    image = Image.open(image_path)
    ori_size = image.size
    image = image.resize((1216, 352))
    image = np.asarray(image, dtype=np.float32) / 255.0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if dataset_args.do_kb_crop is True:
        height, width = image.shape[0], image.shape[1]
        top_margin = height - 352
        left_margin = int((width - 1216) / 2)
        image = image[top_margin: top_margin + 352, left_margin: left_margin + 1216, ...]
    image = np.expand_dims(image, 0)  # [b, h, w, c]
    image = image.transpose((0, 3, 1, 2))
    image = torch.from_numpy(image)
    image = normalize(image)
    return image.cuda() if torch.cuda.is_available() else image, ori_size

def prepare_model(ckpt_path):
    model = Attn2AttnDepth(args)
    model = nn.DataParallel(model)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    print('load ckpt success...')
    return model.eval().cuda() if torch.cuda.is_available() else model.eval()

def batch_predict(dir_path, ckpt_path, save_path):
    model = prepare_model(ckpt_path)
    paths = os.listdir(dir_path)
    paths = sorted(paths)
    for i in tqdm(range(len(paths))):
        image_path = os.path.join(dir_path, paths[i])
        pro_image, ori_size = process_image(image_path)
        pred_image = predict_one_image(model, pro_image)
        pred_image = cv2.resize(pred_image, ori_size)
        cur_save_path = os.path.join(save_path, paths[i])
        plt.imsave(cur_save_path, np.log10(pred_image), cmap='plasma_r')
    print('over...')

if __name__ == '__main__':
    image_path = r'frame/frame_000007.png'
    ckpt_path = r'kitti_without_mosaic/PDepth/model-131835-best_abs_rel_0.06169'
    # pro_image, ori_size = process_image(image_path)
    # model = prepare_model(ckpt_path)
    # pred_image = predict_one_image(model, pro_image)
    # pred_image = cv2.resize(pred_image, ori_size)
    # print('type:', type(pred_image))
    # print(pred_image.shape)
    # plt.imsave('t2.png', np.log10(pred_image), cmap='plasma_r')
    batch_predict('frame', ckpt_path, 'plt_image')

