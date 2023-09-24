from argparse import ArgumentParser
import os, io

import urllib.request

from fairseq.data import Dictionary
import torch

try:
    from .model import create_TrOCR_model
    from .bpe import GPT2BPE
    from .dataset import STRDataset
    from .process import Preprocess, process_model
except ImportError:
    from model import create_TrOCR_model
    from bpe import GPT2BPE
    from dataset import STRDataset
    from process import Preprocess, process_model

def load_dict(dict_path_or_url):
    if dict_path_or_url is not None and dict_path_or_url.startswith('https') :
        content = urllib.request.urlopen(dict_path_or_url).read().decode()
        dict_file = io.StringIO(content)
        tgt_dict = Dictionary.load(dict_file)
    elif os.path.exists(dict_path_or_url):
        content = io.StringIO(dict_path_or_url)
        tgt_dict = Dictionary.load(dict_path_or_url)
    else:
        raise ValueError('Could not find tgt_dictionary.')
    
    return tgt_dict

def main(args):
    if hasattr(args.model_path) and args.model_path is not None:
        process_model(args.model_path)
    
    tgt_dict = load_dict()
    model_args = torch.load('./state.pt')
    model = create_TrOCR_model(model_args)

def cli():
    parser = ArgumentParser()
    parser.add_argument('--adapter-path', default=None, type=str, help='If using adapter or not.')
    parser.add_argument('--image-path', type=str, required=True, help='The image to inference.')
    parser.add_argument('--output-path', default='./result', type=str, help='Path to store result.')
    parser.add_argument('--tgt-dict', default='./gpt2_with_mask.dict.txt', type=str, help='Target dictionary.')
    
    if not os.path.exists('./encoder.pt') and not os.path.exists('./decoder.pt'):
        parser.add_argument('--model-path', default='./trocr-base.pt', type=str, help='Original trocr model path.')
    else:
        parser.add_argument('--state-path', default='./state.pt', type=str, help='States path.')
        parser.add_argument('--encoder-path', default='./encoder.pt', type=str, help='Encoder path.')
        parser.add_argument('--decoder-path', default='./decoder.pt', type=str, help='Decoder path.')
        
    main(parser.parse_args())

if __name__ == '__main__':
    cli()