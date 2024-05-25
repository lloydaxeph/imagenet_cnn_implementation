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
Download and upack the dataset and change the `imagenet_data_dir` variable in [config.py](https://github.com/lloydaxeph/imagenet_cnn_implementation/blob/master/config.py)
```
imagenet_data_dir = '...\ImageNet-Mini'
```

### 2.3 Training Configs
All other important configs can also be found in [config.py](https://github.com/lloydaxeph/imagenet_cnn_implementation/blob/master/config.py). Change it as you would like.
```
epochs = 3
batch_size = 64
learning_rate = 0.001
```

### 2.4 Trigger Training
To trigger training, simply input the following in your terminal
```
python3 train.py
```
