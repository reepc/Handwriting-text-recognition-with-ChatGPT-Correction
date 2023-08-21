import torch, os
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torchvision.transforms import transforms

from tqdm.auto import tqdm

from PIL import Image
from data_preprocess import Image_preprocess

url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
image = Image.open("./362282794_1591132984745448_7946530766554237278_n.jpg").convert("RGB")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class image_dataset(Dataset):
    def __init__(self, path, files = None):
        super(image_dataset).__init__()
        self.path = path
        if files != None:
            self.files = files
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file_name = self.files[index]
        image = Image.open(file_name)

        return image

train_set = DataLoader(image_dataset(), batch_size = 5, shuffle = True)

class image2text(nn.Module):
    def __init__(self, processor, model, adapter):
        super(image2text, self).__init__()
    
    def forward(self, x):
        pass

class Adapter(nn.Module):
    """Create Adapter"""
    def __init__(self):
        super(Adapter, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(),
            nn.ReLU()
        )
    
    def forward(self, x):
        out = self.adapter(x)
        return out

def save_adapter(adapter, name):
    if os.path.isdir("./model"):
        save_path = f"./model/{name}.pt"
        torch.save(adapter.state_dict(), save_path)
    else:
        os.makedirs('models')