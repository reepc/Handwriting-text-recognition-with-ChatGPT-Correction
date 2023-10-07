import cv2
import numpy as np
from deskew import determine_skew

try:
    from .Line_removal import Line_Removal as LR
except ImportError:
    from Line_removal import Line_Removal as LR


def horizontal_projection(binary):
    h, w = binary.shape
    project_img = np.zeros(shape=(binary.shape), dtype=np.uint8) + 255
    for i in range(h):
        num = 0
        for j in range(w):
            if binary[i][j] == 0:
                num+=1
                
        for k in range(num):
            project_img[i][k] = 0
    
    cv2.imwrite('projected.jpg', project_img)
    
def unshadow(img):
    rgb = cv2.split(img)
    result = []
    norm = []
    
    for i in rgb:
        dilated = cv2.dilate(i, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated, 21)
        diff_img = 255 - cv2.absdiff(i, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result.append(diff_img)
        norm.append(norm_img)
    
    result = cv2.merge(result)
    norm = cv2.merge(norm)
    
    return norm

def deskew(img):
    height, width = img.shape[:2]
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    
    rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated

if __name__ == "__main__":
    img_path = '../origin/IMG_2094.jpg'
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    unshadowed = unshadow(gray)
    remover = LR(unshadowed)
    distance = remover.line_distance()
    line_start, line_end = remover.find_lines(distance)
    removerd_pixels = remover.line_iterator(line_start, line_end, window_start=3, window_end=6)