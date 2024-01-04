### The code of  "Conv-Transformer Fusion: Monocular Depth Estimation with Convolutions and Transformers"

[toc]



#### How to use this repo

*   create a virtual environment by conda and activate it

    `conda create -n depth python=3.8`

    `conda activate depth`

*   Installation of dependencies

​		`pip install -r requirements.txt`

*   prepare dataset

​		download the KITTI dataset and NYU-Depth V2 dataset from [here]([The KITTI Vision Benchmark Suite (cvlibs.net)](https://www.cvlibs.net/datasets/kitti/raw_data.php)) and [here]([NYU Depth V2 « Nathan Silberman](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html#raw_parts))

*   After dataset preparation, you need to set different configure parameters to fit different datasets, the configure file is in `utile/opt.py`

*   train 

​		run`python train.py`

