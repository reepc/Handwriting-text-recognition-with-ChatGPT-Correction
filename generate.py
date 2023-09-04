from argparse import ArgumentParser
import os

try:
    from .model import create_TrOCR_model
    from .bpe import GPT2BPE
    from .data_preprocess import STRDataset
except ImportError:
    from model import create_TrOCR_model
    from bpe import GPT2BPE
    from data_preprocess import STRDataset

def main(args):
    pass

def cli():
    parser = ArgumentParser()
    parser.add_argument('--adapter-path', default=None, type=str, help='If using adapter or not.')
    parser.add_argument('--image-path', type=str, required=True, help='The image to inference.')
    parser.add_argument('--output-path', default='./result', type=str, required=True, help='Path to store result')
    if not os.path.exists('./encoder.pt') and not os.path.exists('./decoder.pt'):
        parser.add_argument('--model-path', default='./trocr-base.pt', type=str, required=True, help='Where trocr model is.')
    else:
        parser.add_argument('--state-path', default='./state.pt', type=str, help='Where state is.')
        parser.add_argument('--encoder-path', default='./encoder.pt', type=str, help='Where encoder is.')
        parser.add_argument('--decoder-path', default='./decoder.pt', type=str, help='Where decoder is.')
        
    main(parser.parse_args())

if __name__ == '__main__':
    cli()