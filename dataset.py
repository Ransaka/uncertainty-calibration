import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ISICDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(256, 256), image_ids=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.image_ids = image_ids if image_ids else [
            f.replace('.jpg', '') for f in os.listdir(image_dir) if f.endswith('.jpg')
        ]
        
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{img_id}_segmentation.png")
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # Grayscale
        
        image = self.transform(image)
        mask = self.mask_transform(mask)
        
        # Binarize mask
        mask = (mask > 0.5).float()
        
        return {
            'image': image,
            'mask': mask,
            'id': img_id
        }
