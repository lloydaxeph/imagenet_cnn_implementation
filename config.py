import torch
from torchvision import transforms

# Training Configs -----------------------------------------------------------------------------------------------------
epochs = 100
batch_size = 64
learning_rate = 0.001
imagenet_data_dir_train = r'mini-imagenet\ImageNet-Mini'
imagenet_data_dir_test = r'mini-imagenet\ImageNet-Mini'
val_split_percent = 0.2

shuffle = True  # DataLoader.shuffle
num_workers = 4  # DataLoader.num_workers
n_classes = 1000
loss_label_smoothing = 0.1
transform = transforms.Compose([
  transforms.Resize((224, 224)),  # Resize to a common size (adjust as needed)
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ImageNet models
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ----------------------------------------------------------------------------------------------------------------------
