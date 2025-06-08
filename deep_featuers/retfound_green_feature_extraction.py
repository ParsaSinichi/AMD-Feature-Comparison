import torch
import timm
from torchvision import transforms
from PIL import Image
import yaml
with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = config['rfg_model_path']
rfg = timm.create_model('vit_small_patch14_reg4_dinov2', img_size=(392, 392), num_classes=0)
rfg.load_state_dict(torch.load(model_path, map_location=device))
rfg.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((392, 392)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def rfg_extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = rfg(tensor)
    return features.squeeze(0).cpu().numpy()