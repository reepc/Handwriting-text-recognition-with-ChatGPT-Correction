from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os

""" class IAM_dataset(Dataset):
    def __init__(self, image_path, file_path) -> None:
        super().__init__()
        self.transform = transforms.PILToTensor()
        
        self.sentences = []
        self.imgs = []
        i = 0
        with open(file_path, mode='r') as rf:
            for line in rf:
                line = line.split('\t')
                self.imgs.append(os.path.join(image_path, line[0]))
                self.sentences.append(line[1].replace('\n', ''))
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        imgs_name = self.imgs[index]
        img = Image.open(imgs_name).convert('RGB')
        img = self.transform(img)
        sentence = self.sentences[index]
        
        return img, sentence """

class IAM_dataset:
    def __init__(self, image_path, file_path) -> None:
        self.transform = transforms.PILToTensor()
        
        self.sentence = []
        self.imgs = []
        with open(file_path, mode='r') as rf:
            for line in rf:
                line = line.split('\t')
                self.imgs.append(os.path.join(image_path, line[0]))
                self.sentence.append(line[1].replace('\n', ''))
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        imgs_name = self.imgs[index]
        img = Image.open(imgs_name).convert('RGB')
        sentence = self.sentence[index]
        
        return img, sentence

set = IAM_dataset(image_path='./IAM/image', file_path='./IAM/gt_test.txt')

i = 0
for img, sentence in tqdm(set):
    print(sentence)
    i+=1
    if i == 1:
        break