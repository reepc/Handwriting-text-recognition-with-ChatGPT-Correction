from argparse import ArgumentParser
from tqdm.auto import tqdm
import logging
import os, io

import urllib.request

from fairseq import utils
from fairseq.data import Dictionary
from torch.utils.data import DataLoader
import torch

try:
    from .bpe import GPT2BPE
    from .dataset import STRDataset
    from .model import create_TrOCR_model
    from .ChatGPT_correction import Correction
    from .Preprocess.main import processing, process_model
except ImportError:
    from bpe import GPT2BPE
    from dataset import STRDataset
    from model import create_TrOCR_model
    from Preprocess.main import processing, process_model

def set_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(filename='./model.log', encoding='utf-8', mode='w')
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

logger = set_logger()

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
    
    tgt_dict = load_dict(args.tgt_dict)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_args = torch.load('./state.pt')
    model = create_TrOCR_model(model_args, tgt_dict)
    model.eval().to(device)
    
    processing(args.img_path)
    dataset = STRDataset(img_dir='./splited')
    data = DataLoader(dataset, shuffle=False)
    
    bpe = GPT2BPE()
    doc_list = []
    with torch.no_grad():
        for img in tqdm(data):
            outs = model(img.to(device))
            outs = outs[0][0]
            
            tokens, strs, aligment = utils.post_process_prediction(
                hypo_tokens=outs['tokens'].int().cpu(),
                src_str='',
                alignment=outs['alignment'],
                align_dict=None,
                tgt_dict=model.tgt_dictionary,
                remove_bpe=None,
                extra_symbols_to_ignore={2}
            )
            
            doc_list.append(bpe.decode(strs))
    
    doc = '\n'.join(doc_list)
    try:
        if args.prompt.endswith('.txt'):
            with open(args.prompt, mode='r') as prompt_file:
                prompt = prompt_file.readlines()
                prompt = '\n'.join(prompt)
        
            corrected = Correction(prompt).correct(doc)
        else:
            corrected = Correction(args.prompt).correct(doc)
            full_doc = corrected
    except:
        full_doc = doc
    
    with open(f'{args.output_path}', mode='a') as out_file:
        out_file.write(full_doc)
    
    logger.info(f"Recognized document: '\n' {full_doc}")

def cli():
    parser = ArgumentParser()
    parser.add_argument('--image-path', type=str, required=True, help='The image to inference.')
    parser.add_argument('--prompt', type=str, help='The prompt that you want to sent to ChatGPT, can be a txt file or just type here.')
    parser.add_argument('--output-path', default='./result/output_doc.txt', type=str, help='Path to store result, must end with .txt.')
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