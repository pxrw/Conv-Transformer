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
        mosaic = self.mode == 'train'
        line = self.filenames[idx]

        if mosaic:
            lines = random.sample(self.filenames, 3)
            lines.append(line)
            image, depth = self.get_random_data_with_mosaic(lines, self.args.input_height, self.args.input_width)
            image, depth = self.train_preprocess(image, depth)
            sample = {'image': image, 'depth': depth, 'focal': self.focal}
        else:
            sample = self.get_random_data_for_eval(line, self.args.input_height, self.args.input_width)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_random_data_with_mosaic(self, lines, input_height, input_width):
        '''
        对输入数据的马赛克数据增强方法
        通过可视化可发现，在gt数据的上半部分没有点，所以进行了数据裁切，将所有的输入数据都先处理到高度为224的尺度
        然后采用和yolov4类似的马赛克数据增强方法进行处理
        '''
        min_offset_x, min_offset_y = self.rand(), self.rand()
        image_datas, depth_datas = [], []
        idx = 0

        for line in lines:
            divided_line = line.split()
            if self.args.dataset == 'kitti':
                rgb_file = divided_line[0]
                depth_file = divided_line[1]
                if self.args.use_right is True and random.random() > 0.5:
                    rgb_file.replace('image_02', 'image_03')
                    depth_file.replace('image_02', 'image_03')
            else:
                rgb_file = divided_line[0]
                depth_file = divided_line[1]

            image_path = os.path.join(self.args.data_path, rgb_file).replace('annotation/train/', 'image/')
            depth_path = os.path.join(self.args.data_path, depth_file)

            image = Image.open(image_path)
            depth_gt = self.open_depth_file(depth_path)

            # 对数据进行裁切（目的是裁掉深度gt上半没有数据的部分）
            if self.args.do_kb_crop is True:
                height, width = image.height, image.width
                top_margin = height - input_height
                left_margin = int((width - 1216) / 2)
                image = image.crop((left_margin, top_margin, left_margin + 1216, height))
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, height))
            # print('ori depth_gt max:', np.max(np.asarray(depth_gt)))   # 这里没问题

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

            # 这里需要注意，深度图的数值范围不是0-255，如果使用'L'方法，会new出8-bit，'F'对应32-bit
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
        depth = np.asarray(mosaic_depth, dtype=np.float32)
        depth = np.expand_dims(depth, axis=2)
        depth /= 256.0

        if image.shape[0] != self.args.input_height or image.shape[1] != self.args.input_width:
            image, depth = self.random_crop(image, depth, self.args.input_height, self.args.input_width)

        return image, depth


    def get_random_data_for_eval(self, line, input_height, input_width):
        if self.mode == 'online_eval':
            data_path = self.args.data_path_eval
        else:
            data_path = self.args.data_path

        rgb_file = line.replace('\n', '')

        image_path = os.path.join(data_path, rgb_file)
        image = Image.open(image_path)
        image = np.asarray(image, dtype=np.float32) / 255.0

        if self.mode == 'online_eval':
            depth_path = image_path.replace('/image/', '/groundtruth_depth/').replace('sync_image', 'sync_groundtruth_depth')
            has_valid_depth = False
            depth_gt = self.open_depth_file(depth_path)
            if isinstance(depth_gt, Image.Image):
                has_valid_depth = True

            if has_valid_depth is True:
                depth_gt = np.asarray(depth_gt, dtype=np.float32)
                depth_gt = np.expand_dims(depth_gt, axis=2)
                if self.args.dataset == 'nyu':
                    depth_gt = depth_gt / 1000.0
                else:
                    depth_gt = depth_gt / 256.0

        if self.args.do_kb_crop is True:
            height, width = image.shape[0], image.shape[1]
            top_margin = height - input_height
            left_margin = int((width - 1216) / 2)
            image = image[top_margin: top_margin + input_height, left_margin: left_margin + 1216, ...]

            if self.mode == 'online_eval' and has_valid_depth is True:
                depth_gt = depth_gt[top_margin: top_margin + input_height, left_margin: left_margin + 1216, ...]

        if self.mode == 'online_eval':
            sample = {'image': image, 'depth': depth_gt, 'focal': self.focal, 'has_valid_depth': has_valid_depth}
        else:
            sample = {'image': image, 'focal': self.focal}
        return sample

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

    def rand(self):
        return np.random.rand()

    def open_depth_file(self, path):
        try:
            depth_gt = Image.open(path)
        except IOError:
            print('no matching depth file...')
            depth_gt = False
        return depth_gt


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

    dl = AttnDataLoader(args, mode='train')
    print(len(dl.data))

    sample_batch = next(iter(dl.data))
    image = sample_batch['image'][0].numpy()
    depth_gt = sample_batch['depth'][0].numpy()

    print('image shape:', image.shape)
    print('depth shape:', depth_gt.shape)
    print(np.max(image))
    print(np.max(depth_gt))
