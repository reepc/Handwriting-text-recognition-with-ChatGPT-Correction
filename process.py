import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from deskew import determine_skew

import torch

import math

class Preprocess:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.height, self.width = self.image.shape[:2]
        """ self.image = cv2.resize(self.image, (int(self.width * 0.5), int(self.height * 0.5)))
        cv2.imwrite('./resized.jpg', self.image)
        self.width, self.height = int(self.width * 0.5), int(self.height * 0.5) """
        
    def process(self):
        rotated_image = self.rotate()
        gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
        retval, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        h = self.getHProjection(threshed)
        start = False
        h_start = []
        h_end = []
        
        for index, element in enumerate(h):
            if element > 50 and not start:
                h_start.append(index)
                start = True
                
            if element <= 50 and start:
                h_end.append(index)
                start = False
        
        for i in range(len(h_start)):
            cv2.imwrite(f'./splited2/splited_{i}.jpg', rotated_image[h_start[i]:h_end[i]])
            
        """ hist = cv2.reduce(threshed, 1, cv2.REDUCE_AVG).reshape(-1)
        th = 2
        uppers = [y for y in range(self.height - 1) if hist[y] <= th and hist[y+1] > th]
        lowers = [y for y in range(self.height - 1) if hist[y] > th and hist[y+1] <= th]
        new_uppers = []
        new_lowers = []
        
        for i in range(len(uppers)):
            if abs(uppers[i] - lowers[i]) <= 20:
                continue
            elif abs(uppers[i] - lowers[i]) >= 80:
                half = (lowers[i] - uppers[i]) // 2
                new_uppers.append(uppers[i])
                new_uppers.append(uppers[i] + half + 1)
                
                new_lowers.append(lowers[i] - half - 1)
                new_lowers.append(lowers[i])
            else:
                new_uppers.append(uppers[i])
                new_lowers.append(lowers[i])
            
            ex.
            lowers: [10, 20, 30]  -> [10, 16.5, 20, 30]
            uppers: [5, 15, 25] -> [5, 15, 17.5, 25]
           
            
            if i == len(lowers) - 1:
                break
        
        if new_uppers[0] > new_lowers[0]:
            new_uppers, new_lowers = new_lowers, new_uppers
        
        for j in range(len(new_uppers)):
            distance = abs(new_uppers[j] - new_lowers[j])
            
            if distance <= 40:
                half_dis = distance // 2
                cv2.imwrite(f'./splited2/pic_{j}.jpg', rotated_image[new_uppers[j]-half_dis:new_lowers[j]+half_dis])
            
            else:
                cv2.imwrite(f'./splited2/pic_{j}.jpg', rotated_image[new_uppers[j]:new_lowers[j]])
        
        cv2.imwrite(f'./splited2/pic_{j + 1}.jpg', rotated_image[new_lowers[-1]:])  """
            
    def getHProjection(self, image):
        hProjection = np.zeros(image.shape, np.uint8)
        (h, w)=image.shape
        h_ = [0] * h

        for y in range(h):
            for x in range(w):
                if image[y, x] == 255:
                    h_[y] += 1

        for y in range(h):
            for x in range(h_[y]):
                hProjection[y, x] = 255
        
        return h_
    
    def rotate(self):
        grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        angle = determine_skew(grayscale)
        
        image_center = tuple(np.array(self.image.shape[1::-1]) / 2)
        rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        
        rotated = cv2.warpAffine(self.image, rotation_matrix, (self.width, self.height))
        cv2.imwrite('./rotated.jpg', rotated)
        return rotated

    def binarization(self):
        bin_img = cv2.adaptiveThreshold(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        """ denoise_img = cv2.fastNlMeansDenoising(self.grayscale, None, 30, 7, 15) """
        return bin_img
    
    def denoise(self):
        pass
    
    def extract_text(self):
        pass

def process_model(model_path):
    ost = torch.load(model_path)['model']
    encoder_state = {}
    decoder_state = {}

    for key, value in ost.items():
        new_key = key.replace('encoder.','').replace('decoder.','')
        if key.startswith('encoder.'):
            encoder_state[new_key] = value
        else:
            decoder_state[new_key] = value

    torch.save(encoder_state, 'encoder.pt')
    torch.save(decoder_state, 'decoder.pt')

    torch.save(torch.load(model_path)['cfg']['model'], 'state.pt')
    
if __name__ == "__main__":
    img_path = './image/IMG_2036.jpg'
    processor = Preprocess(img_path)
    """ cv2.imwrite('./reduces.jpg', processor.binarization()) """
    processor.process()