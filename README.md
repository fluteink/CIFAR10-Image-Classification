# CIFAR10-Image-Classification

原作者Github地址：https://github.com/facebookresearch/LaMCTS/tree/main/LaNAS/LaNet

## 配置环境
尽量使用conda环境，以下环境经过测试可以运行
```
conda create -n lanet python=3.7 -y
conda activate lanet
pip install git+https://github.com/ildoonet/cutmix
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install torchviz
conda install scikit-learn
conda install tensorboard
conda install tensorboardx
conda install numpy==1.19.5
```

## 训练 LaNet
1.在data文件夹中下载并解压CIFAR10数据集，数据存放格式如下，下载地址：https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```
data
│── batches_meta
│── data_batch_1
│── data_batch_2
│── data_batch_3
│── data_batch_4
│── data_batch_5
│── readme.html 
│── test_batch
```

2. 运行以下命令进行训练，根据自己需求更改训练文件名，epochs参数，注意在A800-80G上训练一个epoch需要10分钟左右，train中的优化器是作者默认设置的SGD优化器，其他优化器文件设置见文件名

```
cd 你的文件夹存放路径/CIFAR10
mkdir checkpoints
python train.py --auxiliary --batch_size=32 --init_ch=128 --layer=24 --arch='[2, 2, 0, 2, 1, 2, 0, 2, 2, 3, 2, 1, 2, 0, 0, 1, 1, 1, 2, 1, 1, 0, 3, 4, 3, 0, 3, 1]' --model_ema --model-ema-decay 0.9999 --auto_augment --epochs 1500
```

```[2, 2, 0, 2, 1, 2, 0, 2, 2, 3, 2, 1, 2, 0, 0, 1, 1, 1, 2, 1, 1, 0, 3, 4, 3, 0, 3, 1]``` 是作者用分布式GPU搜索出来的最佳网络结构，具体快照如下

<p align="center">
<img src='https://github.com/linnanwang/paper-image-repo/blob/master/LaNAS/distributed_search_results.png?raw=true' width="600">
</p>

## 测试LaNet

1. 可以从 <a href="https://drive.google.com/file/d/1bZsEoG-sroVyYR4F_2ozGLA5W50CT84P/view?usp=sharing">这里</a>下载官方作者的预训练权重文件, 然后解压并将它替换到CIFAR10文件的checkpoint文件夹中

2. 运行以下命令进行测试.
```
python test.py  --checkpoint  ./lanas_128_99.03 --layers 24 --init_ch 128 --arch='[2, 2, 0, 2, 1, 2, 0, 2, 2, 3, 2, 1, 2, 0, 0, 1, 1, 1, 2, 1, 1, 0, 3, 4, 3, 0, 3, 1]'
```
