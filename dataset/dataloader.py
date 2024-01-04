from torch.utils import data
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
import random, os, cv2
import numpy as np
from torchvision import transforms
from dataset.transform_list import ToTensor
from dataset.distributed_sampler_no_evenly_divisible import DistributedSamplerNoEvenlyDivisible

ImageFile.LOAD_TRUNCATED_IMAGES = True

def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode='train', transform=None, is_for_online_eval=False):
        super(DataLoadPreprocess, self).__init__()
        self.args = args
        self.mode = mode
        self.focal = 518.8579
        # 读数据
        if mode == 'online_eval':
            with open(self.args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.length = len(self.filenames)
        self.transform = transform
        self.is_for_online_eval = is_for_online_eval

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = idx % self.length
        sample_path = self.filenames[idx]
        self.focal = 518.8579

        if self.mode == 'train':
            image, depth = self.get_random_data_for_train(sample_path, self.args.input_height, self.args.input_width)
            image, depth = self.train_preprocess(image, depth)
            sample = {'image': image, 'depth': depth, 'focal': self.focal}
        else:
            sample = self.get_random_data_for_eval(sample_path, self.args.input_height)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_random_data_for_train(self, sample_path, input_height, input_width):
        '''
        训练过程中的数据随机增强，使用了以下方法：
        裁剪：将数据裁剪到[352, 1216]的尺度
        旋转：随机旋转角度
        随机裁剪：如果当前[352, 1216]的尺度仍然大于给定的输入尺度[352, 1120]，将将其随机裁剪到这个尺度上
        随机翻转：左右翻转
        随机光照增强
        '''
        divided_file = sample_path.split()
        rgb_file = divided_file[0]
        depth_file = divided_file[1]
        if self.args.use_right is True and random.random() > 0.5:
            rgb_file.replace('image_02', 'image_03')
            depth_file.replace('image_02', 'image_03')

        image_path = os.path.join(self.args.data_path, rgb_file).replace('annotation/train/', 'image/')
        depth_path = os.path.join(self.args.data_path, depth_file)

        image = Image.open(image_path)
        depth_gt = self.open_depth_file(depth_path)

        if self.args.do_kb_crop is True:
            # args.width = 1120 < 1216
            height, width = image.height, image.width
            top_margin = int(height - input_height)
            left_margin = int((width - 1216) / 2)
            depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + input_height))
            image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + input_height))  # Image

        if self.args.do_random_rotate is True:
            random_angle = (random.random() - 0.5) * 2 * self.args.degree
            image = self.rotate_image(image, random_angle)
            depth_gt = self.rotate_image(depth_gt, random_angle)

        # 归一化处理 Image -> ndarray
        image = np.asarray(image, dtype=np.float32) / 255.0
        depth_gt = np.asarray(depth_gt, dtype=np.float32)
        depth_gt = np.expand_dims(depth_gt, axis=2)
        depth_gt /= 256.0   # 因为给定的args.max_depth = 80，只有除以256.0才能让depth_gt的最大值在80附近，其实还是有超过80的

        if image.shape[0] != input_height or image.shape[1] != input_width:
            image, depth_gt = self.random_crop(image, depth_gt, input_height, input_width)

        return image, depth_gt

    def get_random_data_for_eval(self, sample_path, input_height):
        '''
        验证时的数据读取
        只需要将数据裁剪到[352, 1216]这个尺度上即可
        '''
        if self.mode == 'online_eval':
            data_path = self.args.data_path_eval
        else:
            data_path = self.args.data_path

        rgb_file = sample_path.replace('\n', '')
        # depth_file = divided_file[1]
        image_path = os.path.join(data_path, rgb_file)
        print(image_path)
        image = Image.open(image_path)
        image = np.asarray(image, dtype=np.float32) / 255.0

        if self.mode == 'online_eval':
            depth_path = image_path.replace('/image/', '/groundtruth_depth/').replace('sync_image', 'sync_groundtruth_depth')
            print(depth_path)
            has_valid_depth = False
            depth_gt = self.open_depth_file(depth_path)
            if isinstance(depth_gt, Image.Image):
                has_valid_depth = True

            if has_valid_depth is True:
                depth_gt = np.asarray(depth_gt, dtype=np.float32)
                depth_gt = np.expand_dims(depth_gt, axis=2)
                depth_gt /= 256.0

        if self.args.do_kb_crop is True:
            height, width = image.shape[0], image.shape[1]
            top_margin = int(height - input_height)
            left_margin = int((width - 1216) / 2)
            image = image[top_margin: top_margin + input_height, left_margin: left_margin + 1216, ...]
            if self.mode == 'online_eval' and has_valid_depth:
                depth_gt = depth_gt[top_margin: top_margin + 352, left_margin: left_margin + 1216, ...]

        if self.mode == 'online_eval':
            # image: [h, w, c]
            sample = {'image': image, 'depth': depth_gt, 'focal': self.focal, 'has_valid_depth': has_valid_depth}
        else:
            sample = {'image': image, 'focal': self.focal}

        return sample

    def train_preprocess(self, image, depth_gt):
        do_flip = random.random()
        if do_flip > 0.5:
            # h, w, c -> h, w[::-1], c，相当于做了图像的左右翻转
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def open_depth_file(self, path):
        try:
            depth_gt = Image.open(path)
        except IOError:
            print('no matching depth file...')
            depth_gt = False
        return depth_gt

    def rotate_image(self, image, angle, flag=Image.Resampling.BILINEAR):
        return image.rotate(angle, resample=flag)

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y: y + height, x: x + width, ...]
        depth = depth[y: y + height, x: x + width, ...]
        return img, depth

    def augment_image(self, image):
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

class AttnDataLoader:
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.train_sampler = data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None

            self.data = data.DataLoader(self.training_samples, args.batch_size, shuffle=(self.train_sampler is None),
                                        num_workers=args.num_threads, pin_memory=True, sampler=self.train_sampler)
        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            self.data = data.DataLoader(self.testing_samples, batch_size=1, shuffle=False, num_workers=1,
                                        pin_memory=True, sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = data.DataLoader(self.testing_samples, batch_size=1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))

if __name__ == '__main__':
    from utils.opt_kitti import args

    dl = AttnDataLoader(args, mode='online_eval')
    print(len(dl.data))
    sample_batch = next(iter(dl.data))
    image = sample_batch['image'][0].numpy()
    depth_gt = sample_batch['depth'][0].numpy()

    print('image shape:', image.shape)
    print('depth shape:', depth_gt.shape)
    print(np.max(image))
    print(np.max(depth_gt))


# /media/data3/pxrdata/KITTI/selection/depth_selection/val_selection_cropped/image