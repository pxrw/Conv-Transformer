import torch
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
    def __init__(self, args, mode, transform=None, is_for_online_eval=False, mosaic=False,
                 mosaic_prob=0.5, special_aug_ratio=0.7):
        super(DataLoadPreprocess, self).__init__()
        self.args = args.nyu
        self.mode = mode
        self.focal = 518.8579
        self.epoch_length = args.num_epochs
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.special_aug_ratio = special_aug_ratio
        self.epoch_now = -1

        if mode == 'train':
            with open(self.args.filenames_file, 'r') as f:
                self.filenames = f.readlines()
        elif mode == 'online_eval':
            with open(self.args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(self.args.filenames_file_test, 'r') as f:
                self.filenames = f.readlines()

        self.length = len(self.filenames)
        self.transform = transform
        self.is_for_online_eval = is_for_online_eval

    def __len__(self):
        return self.length

    def rand(self):
        return np.random.rand()

    def __getitem__(self, idx):
        line = self.filenames[idx]
        if self.mode == 'train':
            if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
                lines = random.sample(self.filenames, 3)
                lines.append(line)
                random.shuffle(lines)
                image, depth = self.get_random_data_with_mosaic(lines, self.args.input_height, self.args.input_width)
            else:
                image, depth = self.get_random_data(line, self.args.input_height, self.args.input_width)

            image, depth = self.train_preprocess(image, depth)
            sample = {'image': image, 'depth': depth, 'focal': self.focal}
        else:
            sample = self.get_random_data_for_eval(line, self.args.input_height, self.args.input_width)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


    def get_random_data_with_mosaic(self, lines, input_height, input_width):
        min_offset_x, min_offset_y = self.rand(), self.rand()
        image_datas, depth_datas = [], []
        idx = 0

        for line in lines:
            divided_line = line.split()
            rgb_file = divided_line[0]
            depth_file = divided_line[1]

            image_path = self.args.data_path + rgb_file
            depth_path = self.args.gt_path + depth_file

            image = Image.open(image_path)
            depth_gt = self.open_depth_file(depth_path)

            if input_height == 480:
                depth_gt = np.array(depth_gt)
                valid_mask = np.zeros_like(depth_gt)
                valid_mask[45:472, 43:608] = 1
                depth_gt[valid_mask == 0] = 0
                depth_gt = Image.fromarray(depth_gt)
            else:
                depth_gt = depth_gt.crop((43, 45, 608, 472))
                image = image.crop((43, 45, 608, 472))

            if idx == 0:
                dx = int(input_width * min_offset_x) - input_width
                dy = int(input_height * min_offset_y) - input_height
            elif idx == 1:
                dx = int(input_width * min_offset_x) - input_width
                dy = int(input_height * min_offset_y)
            elif idx == 2:
                dx = int(input_width * min_offset_x)
                dy = int(input_height * min_offset_y)
            elif idx == 3:
                dx = int(input_width * min_offset_x)
                dy = int(input_height * min_offset_y) - input_height

            new_image = Image.new('RGB', (input_width, input_height), (0, 0, 0))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            new_depth = Image.new('F', (input_width, input_height), 0)
            new_depth.paste(depth_gt, (dx, dy))
            depth_data = np.array(new_depth)

            idx += 1
            image_datas.append(image_data)
            depth_datas.append(depth_data)

        cutx = int(input_width * min_offset_x)
        cuty = int(input_height * min_offset_y)

        mosaic_image = np.zeros([input_height, input_width, 3])
        mosaic_depth = np.zeros([input_height, input_width])

        mosaic_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        mosaic_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        mosaic_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        mosaic_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        mosaic_depth[:cuty, :cutx] = depth_datas[0][:cuty, :cutx]
        mosaic_depth[cuty:, :cutx] = depth_datas[1][cuty:, :cutx]
        mosaic_depth[cuty:, cutx:] = depth_datas[2][cuty:, cutx:]
        mosaic_depth[:cuty, cutx:] = depth_datas[3][:cuty, cutx:]

        image = np.asarray(mosaic_image, dtype=np.float32) / 255.0
        depth_gt = np.asarray(mosaic_depth, dtype=np.float32)
        depth_gt = np.expand_dims(depth_gt, axis=2)

        depth_gt = depth_gt / 1000.0
        img, depth = image, depth_gt
        H, W = img.shape[0], img.shape[1]
        a, b, c, d = random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)
        l, u = int(a * W), int(b * H)
        w, h = int(max((W - a * W) * c * 0.75, 1)), int(max((H - b * H) * d * 0.75, 1))
        depth_copied = np.repeat(depth, 3, axis=2)
        M = np.ones(img.shape)
        M[l:l + h, u:u + w, :] = 0
        img = M * img + (1 - M) * depth_copied
        image = img.astype(np.float32)

        if image.shape[0] != input_height or image.shape[1] != input_width:
            image, depth_gt = self.random_crop(image, depth_gt, input_height, input_width)

        return image, depth_gt

    def get_random_data(self, line, input_height, input_width):
        divided_file = line.split()
        rgb_file = divided_file[0]
        depth_file = divided_file[1]

        image_path = self.args.data_path + rgb_file
        depth_path = self.args.gt_path + depth_file

        image = Image.open(image_path)
        depth_gt = self.open_depth_file(depth_path)

        if input_height == 480:
            depth_gt = np.array(depth_gt)
            valid_mask = np.zeros_like(depth_gt)
            valid_mask[45:472, 43:608] = 1
            depth_gt[valid_mask == 0] = 0
            depth_gt = Image.fromarray(depth_gt)
        else:
            depth_gt = depth_gt.crop((43, 45, 608, 472))
            image = image.crop((43, 45, 608, 472))

        if self.args.do_random_crop is True:
            random_angle = (random.random() - 0.5) * 2 * self.args.degree
            image = self.rotate_image(image, random_angle)
            depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)

        image = np.asarray(image, dtype=np.float32) / 255.0
        depth_gt = np.asarray(depth_gt, dtype=np.float32)
        depth_gt = np.expand_dims(depth_gt, axis=2)

        depth_gt = depth_gt / 1000.0
        img, depth = image, depth_gt
        H, W = img.shape[0], img.shape[1]
        a, b, c, d = random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)
        l, u = int(a * W), int(b * H)
        w, h = int(max((W - a * W) * c * 0.75, 1)), int(max((H - b * H) * d * 0.75, 1))
        depth_copied = np.repeat(depth, 3, axis=2)
        M = np.ones(img.shape)
        M[l:l + h, u:u + w, :] = 0
        img = M * img + (1 - M) * depth_copied
        image = img.astype(np.float32)

        if image.shape[0] != input_height or image.shape[1] != input_width:
            image, depth_gt = self.random_crop(image, depth_gt, input_height, input_width)

        return image, depth_gt

    def get_random_data_for_eval(self, line, input_height, input_width):
        if self.mode == 'online_eval':
            data_path = self.args.data_path_eval
        else:
            data_path = self.args.data_path_test

        divided_file = line.split()
        rgb_file = divided_file[0]
        depth_file = divided_file[1]

        image_path = os.path.join(data_path, rgb_file)
        image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

        if self.mode == 'online_eval':
            gt_path = os.path.join(data_path, depth_file)
            has_valid_depth = False
            depth_gt = self.open_depth_file(gt_path)
            if isinstance(depth_gt, Image.Image):
                has_valid_depth = True

            if has_valid_depth is True:
                depth_gt = np.asarray(depth_gt, dtype=np.float32)
                depth_gt = np.expand_dims(depth_gt, axis=2)
                depth_gt = depth_gt / 1000.0

        if self.mode == 'online_eval':
            sample = {'image': image, 'depth': depth_gt, 'focal': self.focal, 'has_valid_depth': has_valid_depth}
        else:
            sample = {'image': image, 'focal': self.focal}
        return sample



    def open_depth_file(self, path):
        try:
            depth_gt = Image.open(path)
        except IOError:
                    print('no matching depth file...')
                    depth_gt = False
        return depth_gt

    def rotate_image(self, image, angle, flag=Image.Resampling.BILINEAR):
        return image.rotate(angle, resample=flag)

    def random_crop(self, image, depth, input_height, input_width):
        assert image.shape[0] >= input_height
        assert image.shape[1] >= input_width
        x = random.randint(0, image.shape[1] - input_width)
        y = random.randint(0, image.shape[0] - input_height)
        image = image[y: y + input_height, x: x + input_width, ...]
        depth = depth[y: y + input_height, x: x + input_width, ...]
        return image, depth

    def train_preprocess(self, image, depth):
        do_flip = random.random() > 0.5
        if do_flip:
            image = (image[:, ::-1, :]).copy()
            depth = (depth[:, ::-1, :]).copy()
        do_augment = random.random() > 0.5
        if do_augment:
            image = self.augment_image(image)
        return image, depth

    def augment_image(self, image):
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma
        brightness = random.uniform(0.75, 1.25)
        image_aug = image_aug * brightness

        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)
        return image_aug

class AttnDataLoader:
    def __init__(self, args, mode):
        self.g = torch.Generator().manual_seed(3407)
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.train_sampler = data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None

            self.data = data.DataLoader(self.training_samples, args.batch_size, shuffle=(self.train_sampler is None),
                                        num_workers=args.num_threads, pin_memory=True, sampler=self.train_sampler,
                                        generator=self.g)

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
    from utils.opt import args

    dl = AttnDataLoader(args, mode='train')
    print(len(dl.data))
    # print(dl.training_samples.epoch_now)

    sample_batch = next(iter(dl.data))
    image = sample_batch['image'][0].numpy()
    depth_gt = sample_batch['depth'][0].numpy()

    print('image shape:', image.shape)
    print('depth shape:', depth_gt.shape)
    print(np.max(image))
    print(np.max(depth_gt))

    # with open(args.filenames_file_nyu, 'r') as f:
    #     filenames = f.readlines()
    #
    # sample = filenames[0]
    # rgb_file = sample.split()[0]
    # depth_file = sample.split()[1]
    #
    # print(rgb_file)
    # print(depth_file)
    # print(args.data_path_nyu + rgb_file)
    # try:
    #     Image.open(args.data_path_nyu + rgb_file)
    # except:
    #     raise