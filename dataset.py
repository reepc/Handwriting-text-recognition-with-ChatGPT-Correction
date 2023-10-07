from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

import os

class STRDataset(Dataset):
    def __init__(self, img_dir=None):
        self.images = os.listdir('./splited')
        self.images.sort(key=lambda x:int(x[8:-4]))
        
        self.tfm = transforms.Compose([
            transforms.Resize(size=(384,384), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = Image.open(f"./splited/{self.images[index]}").convert("RGB")
        tfm_img = self.tfm(img)
        
        return tfm_img

if __name__ == "__main__":
    STRDataset()