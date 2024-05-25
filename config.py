import torch
from torchvision import transforms

# Training Configs -----------------------------------------------------------------------------------------------------
epochs = 3
batch_size = 64
learning_rate = 0.001
imagenet_data_dir = r'mini-imagenet\ImageNet-Mini'

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
