import cv2
import numpy as np
from scipy.signal import argrelmin

import math

try:
    from .utils import unshadow, deskew
except ImportError:
    from utils import unshadow, deskew



class segment_1:
    def segment(self, gray):
        """
        Change and improve from https://www.kaggle.com/code/irinaabdullaeva/text-segmentation#Method-#1.
        Original paper: http://ciir.cs.umass.edu/pubfiles/mm-27.pdf.
        """
        transposed = np.transpose(gray)
        window_size = self.compute_window_size(gray)
        kernel = self.create_kernel(9, 4, 1.5)
        filtered = cv2.filter2D(transposed, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        
        normalized = self.normalize(filtered)
        summ = np.sum(normalized, axis=0)
        smoothed = self.smooth(summ, window_size)
        mins = argrelmin(smoothed, order=2)
        arr_mins = np.array(mins)
        
        lines = self.crop_text_to_lines(transposed, arr_mins[0])
        lines_arr = []
        for i in range(len(lines)-1):
            lines_arr.append(np.expand_dims(lines[i], -1))
        
        results = self.transpose_to_normal(lines_arr)
        
        for i in range(len(results)):
            image = cv2.cvtColor(results[i][0], cv2.COLOR_GRAY2BGR)
            cv2.imwrite(f'./splited_{i}.jpg', image)
            
    def compute_window_size(self, gray):
        threshold = np.max(gray) - np.std(gray)
        unshadowed = unshadow(gray)
        binary = np.where(unshadowed > threshold, 255, 0).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        heights = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if h > 20:
                heights.append(h)
            
        window_size = np.average(heights)
        
        return math.ceil(window_size)

    def normalize(self, img):
        (m, s) = cv2.meanStdDev(img)
        m = m[0][0]
        s = s[0][0]
        img = img - m
        img = img / s if s>0 else img
        return img
    
    def smooth(self, x, window_len=11, window='bartlett'):
        if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.") 
        if window_len<3:
            return x
        if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'") 
        s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w = np.ones(window_len,'d')
        else:
            w = eval('np.'+window+'(window_len)')

        y = np.convolve(w/w.sum(),s,mode='valid')
        return y
    
    def crop_text_to_lines(self, text, blanks):
        x1 = 0
        lines = []
        for i,blank in enumerate(blanks):
            x2 = blank
            line = text[:, x1 + 10:x2 + 10]
            lines.append(line)
            x1 = blank
        
        return lines
    
    def transpose_to_normal(self, lines):
        result = []
        for i in lines:
            line = np.transpose(i)
            result.append(line)
        
        return result

    def create_kernel(self, kernel_size, sigma, theta):
        assert kernel_size % 2 # must be odd size
        halfSize = kernel_size // 2

        kernel = np.zeros([kernel_size, kernel_size])
        sigmaX = sigma
        sigmaY = sigma * theta

        for i in range(kernel_size):
            for j in range(kernel_size):
                x = i - halfSize
                y = j - halfSize

                expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
                xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
                yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)

                kernel[i, j] = (xTerm + yTerm) * expTerm

        kernel = kernel / np.sum(kernel)
        return kernel

class Segment_2:
    "TODO: Using NN"
    pass

class Segment_3:
    def segment_3(self, img_path):
        """
        Havent'done
        Reference: https://arxiv.org/ftp/arxiv/papers/2104/2104.08777.pdf
        """
        # Read the image
        img = cv2.imread(img_path)
        rotated = deskew(img)
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        unshadowed = unshadow(gray)
        blurred = cv2.GaussianBlur(unshadowed, (7, 7), 0)
        
        # Binarize the image
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        """ kernel = np.ones((3, 3), np.uint0)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        cv2.imwrite('dilated.jpg', dilated)
         """
        # Detect connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        print(num_labels)
        heights = []
        for i in range(1, num_labels):
            # 绘制每一个连通分量的边界框
            x, y, w, h, area = stats[i]
            heights.append(h)
            cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite('connected.jpg', gray)
        
        avg_height = np.median(heights)
        text_height = self.text_height(rotated.shape)
        
        # Sort blobs based on their y-coordinate for processing
        valid_bboxes = []
        for stat in stats:
            x, y, width, height, area = stat
            if avg_height * 0.5 < height < avg_height * 1.2:
                valid_bboxes.append(stat)
        print(len(valid_bboxes))
        
        # Sort blobs based on their y-coordinate for processing
        valid_bboxes.sort(key=lambda stat: stat[1])

        prev_y = 0
        print(len(valid_bboxes[0]))
        for box in valid_bboxes:
            x, y, w, h = box
            if y - prev_y < text_height:
                pass
            
        assert False
    
    def text_height(self, shape):
        height, width, _ = shape
        height_text = (1/24) * (((height / 2)**2) + (width **2))**0.5
        return height_text

class Segment_4:
    pass