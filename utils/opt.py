from easydict import EasyDict
dic = {
        # 公共参数
        'dataset': 'kitti',
        'mode': 'train',
        'ckpt': '',
        'log_freq': 100,
        'eval_freq': 1000,
        'pretrained': '/home/pxr/pxrProject/DepthEstimation/TransDepth/ckpt/pvt_v2_b5.pth',
        'do_online_eval': True,
        'do_random_rotate': True,
        'log_dir': '/home/pxr/pxrProject/DepthEstimation/DepthWithTransCnn/log',
        'eval_summary_dir': '',
        'save_path': '/home/pxr/pxrProject/DepthEstimation/DepthWithTransCnn/multipatch_kitti_addconv',
        'multiprocessing_distributed': True,
        'dist_url': 'tcp://127.0.0.1:9104',
        'dist_backend': 'nccl',
        'distributed': False,
        'rank': 0,
        'world_size': 1,
        'gpu': '',
        'batch_size': 8,
        'num_epochs': 30,
        'learning_rate': 4e-5,
        'end_learning_rate': -1,
        'weight_decay': 1e-2,
        'adam_eps': 1e-3,
        'variance_focus': 0.85,
        'num_threads': 1,
        'norm': 'BN',
        'act': 'GELU',
        'model_name': 'PDepth',
# ---------------------------------------------------------------------------------------------------------------------
    'kitti': {
        'min_depth': 1e-3,
        'min_depth_eval': 1e-3,
        'max_depth': 80,
        'max_depth_eval': 80,
        'do_random_crop': True,
        'do_random_rotate': True,
        'garg_crop': False,
        'eigen_crop': False,
        'do_kb_crop': True,
        'use_right': False,
        'data_path': "/media/data3/pxrdata/KITTI/annotation/train",
        'data_path_eval': "/media/data3/pxrdata/KITTI/selection",
        'gt_path': '/media/data3/pxrdata/KITTI/annotation',
        'filenames_file': '/home/pxr/pxrProject/DepthEstimation/DepthWithTransCnn/split/kitti_train_file.txt',
        'filenames_file_eval': '/home/pxr/pxrProject/DepthEstimation/DepthWithTransCnn/split/kitti_official_valid.txt',
        'filenames_file_test': '/home/pxr/pxrProject/DepthEstimation/DepthWithTransCnn/split/kitti_official_valid.txt',
        'input_height': 352,
        'input_width': 1120,
        'degree': 1.0,
    },
# ---------------------------------------------------------------------------------------------------------------------
    'nyu': {
        'min_depth': 1e-3,
        'min_depth_eval': 1e-3,
        'max_depth': 10,
        'max_depth_eval': 10,
        'do_random_crop': True,
        'garg_crop': False,
        'eigen_crop': False,
        'do_kb_crop': False,
        'data_path': "/media/data3/pxrdata/nyu",
        'data_path_eval': "/media/data3/pxrdata/nyu",
        'data_path_test': "/media/data3/pxrdata/nyu",
        'gt_path': '/media/data3/pxrdata/nyu',
        'filenames_file': '/home/pxr/pxrProject/DepthEstimation/DepthWithTransCnn/split/nyu_train.txt',
        'filenames_file_eval': '/home/pxr/pxrProject/DepthEstimation/DepthWithTransCnn/split/nyu_test.txt',
        'filenames_file_test': '/home/pxr/pxrProject/DepthEstimation/DepthWithTransCnn/split/nyu_test.txt',
        'input_height': 480,
        'input_width': 640,
        'degree': 1.0,
    }
}

args = EasyDict(dic)

if __name__ == '__main__':
    a = args.nyu
    print(a.filenames_file)

