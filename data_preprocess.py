import cv2
import numpy as np
import tempfile

from torchvision import transforms
from torchvision.transforms import functional
from torchvision.utils import save_image
from fairseq.data import FairseqDataset

from PIL import Image

class Image_preprocess:
    def deskew(self, image):
        co_ords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(co_ords)[-1]
        if angle < -45:
            angle = -(90 + angle)

        else:
            angle = -angle
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated

    def increase_image_ppi(self, image_path):
        img = Image.open(image_path)
        length_x, width_y = img.size
        factor = min(1, float(1024.0 / length_x))
        new_size = int(length_x * factor), int(width_y * factor)
        resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_file_name = temp_file.name
        resized_img.save(temp_file_name, dpi=(300, 300))

class STRDataset(FairseqDataset):
    def __init__(self, img_path):
        self.img_path = img_path
        self.tfm = transforms.Compose([
            transforms.Resize(size=(384,384), interpolation=functional.InterpolationMode.BICUBIC, max_size=None, antialias="warn"),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])
        
    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        img = Image.open(self.img_path).convert("RGB")
        tfm_img = self.tfm(img)
        
        return tfm_img


if __name__ == "__main__":
    tfm = transforms.Compose([
        transforms.Resize(size=(384,384), interpolation=functional.InterpolationMode.BICUBIC, max_size=None, antialias="warn"),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    
    tfm_img = tfm(Image.open('./IAM/image/c04-110-00.jpg').convert('RGB'))
    save_image(tfm_img, 'test.jpg')