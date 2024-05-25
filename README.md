# ImageNet Classification with Deep Convolutional Neural Networks Implementation

## 1.0 About
A simple implementation of the paper [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) using [pytorch](https://pytorch.org/).

### 1.1 Model Architecture
![image](https://github.com/lloydaxeph/imagenet_cnn_implementation/assets/158691653/99685cc8-586c-47a0-9ade-1673847fff88)

## 2.0 Getting Started
### 2.1 Installation
Install the required packages
```
pip3 install -r requirements.txt
```
### 2.2 ImageNet Data
This project is aimed to use the classic [ImageNet Dataset](https://www.image-net.org/). But due to my limited resources, I will just simply use the [Mini-ImageNet Dataset](https://www.kaggle.com/datasets/deeptrial/miniimagenet/data) I found in Kaggle.

### 2.3 Train Dataset
Download and upack the dataset and change the `imagenet_data_dir_train` variable into the dataset's directory in [config.py](https://github.com/lloydaxeph/imagenet_cnn_implementation/blob/master/config.py).
```
imagenet_data_dir_train = '\ImageNet-Mini\train'
```

### 2.3 Trigger Training
To trigger training, simply input the following command in your terminal:
```
python3 train.py --epochs=100 --batch_size=64 --lr=0.001 --val_split=0.2
```
Or you can just edit the parameters in variables in [config.py](https://github.com/lloydaxeph/imagenet_cnn_implementation/blob/master/config.py) and simply use:
```
python3 train.py
```

### 2.5 Test Images
Testing in this project is very simple. You can use the following command for testing where `--model_path` is the path of your pretrained model and `--num_images` is the number of random images from your test dataset:
```
python3 test.py --model_path=mymodel.pt --num_images=10 --print=True
```
Similar to training, you can either input the `--data_path` in your run command or simply set your test dataset's directory into the `imagenet_data_dir_test` variable into the dataset's directory in [config.py].
