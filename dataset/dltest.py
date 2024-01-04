from utils.opt_kitti import args
import os

if __name__ == '__main__':
    with open(args.filenames_file_eval, 'r') as f:
        filenames = f.readlines()

    print(len(filenames))
    sample_path = filenames[0]
    print(sample_path)