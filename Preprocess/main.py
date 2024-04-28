
import cv2

try:
    from .Line_removal import Line_Removal
    from .utils import unshadow, deskew
    from .Segmentation import segment_1
except ImportError:
    from Line_removal import Line_Removal
    from utils import unshadow, deskew
    from Segmentation import segment_1
    
def processing(img_path):
    img = cv2.imread(img_path)
    
    rotated = deskew(img)
    unshadowed = unshadow(rotated)
    
    gray = cv2.cvtColor(unshadowed, cv2.COLOR_BGR2GRAY)
    
    try:
        remover = Line_Removal(gray)
        distance = remover.line_distance()
        line_start, line_end = remover.find_lines(distance)
        removed_pixels = remover.line_iterator(line_start, line_end, window_start=3, window_end=6)
        gray = gray - removed_pixels
    except Exception:
        pass
    
    Segmentor = segment_1()
    Segmentor.segment(gray)

if __name__ == '__main__':
    pass
    