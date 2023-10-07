import torch

import cv2
import os

try:
    from .Line_removal import Line_Removal
    from .Preprocess import *
    from .Segmentation import segment_1
except ImportError:
    from Line_removal import Line_Removal
    from Preprocess import *
    from Segmentation import segment_1

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

    torch.save(encoder_state, '../encoder.pt')
    torch.save(decoder_state, '../decoder.pt')

    torch.save(torch.load(model_path)['cfg']['model'], '../state.pt')
    os.remove(model_path)
    
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
    except:
        pass
    
    Segmentor = segment_1()
    Segmentor.segment(gray)

if __name__ == '__main__':
    process_model('/home/reep_c/Handwriting text recognition with ChatGPT correction/trocr-base.pt')
    