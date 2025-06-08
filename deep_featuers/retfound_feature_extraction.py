import os
import yaml
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import models_vit

with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ImageNet normalization values
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
# image transformation
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])
# No classifier head
retfound_model = models_vit.__dict__['vit_large_patch16'](
    img_size=224,
    num_classes=0,
    drop_path_rate=0,
    global_pool=True
)

checkpoint = torch.load(config["rf_model_path"], map_location='cpu')
_ = retfound_model.load_state_dict(checkpoint['model'], strict=False)
retfound_model.to(device)


def rf_extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = image_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = retfound_model.forward_features(input_tensor)
    return features.squeeze().cpu().numpy()
