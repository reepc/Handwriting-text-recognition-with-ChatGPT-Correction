from fairseq.data import FairseqDataset
from torchvision import transforms
from torchvision.transforms import functional
from torchvision.utils import save_image

from PIL import Image

import os

class STRDataset(FairseqDataset):
    def __init__(self):
        self.images = os.listdir('./splited')
        
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