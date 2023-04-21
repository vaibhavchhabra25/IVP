# Carla Dataset can be downloaded from Kaggle
# For the project, we combined all the data available(5000 images) and splitted it - 4000 for training, 1000 for testing

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os

class CarlaDataset(Dataset):
    def __init__(self, imageDir, maskDir, transform=None):
        self.imageDir = imageDir
        self.transform = transform
        self.maskDir = maskDir
        self.images = os.listdir(imageDir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.imageDir, self.images[index])
        mask_path = os.path.join(self.maskDir, self.images[index]) 
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).split()[0], dtype=np.float32)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        return image, mask
