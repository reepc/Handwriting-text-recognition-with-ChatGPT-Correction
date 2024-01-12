"Haven't done yet"

from fairseq import utils
from fairseq.data import Dictionary

from torch.utils.data import DataLoader
import torch

import pandas as pd

from tqdm.auto import tqdm
import urllib.request
import io, os

try:
    from .model import create_TrOCR_model
    from .bpe import GPT2BPE
    from .dataset import STRDataset
    from .score import CERScorer
except ImportError:
    from model import create_TrOCR_model
    from bpe import GPT2BPE
    from dataset import STRDataset
    from score import CERScorer

import logging

def main(state, data):
    bpe = GPT2BPE()
    model = create_TrOCR_model(state)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.eval().to(device)
    
    scorer = CERScorer(cfg=None)
    for img, text in tqdm(data):
        decoder_out = model.forward(img.to(device))
        
        decoder_out = decoder_out[0][0]
        
        tokens, string, aligment = utils.post_process_prediction(
            hypo_tokens=decoder_out['tokens'].int().cpu(),
            src_str='',
            alignment=decoder_out['alignment'],
            align_dict=None,
            tgt_dict=model.tgt_dictionary,
            remove_bpe=None,
            extra_symbols_to_ignore={2}
        )
        
        detok_str = bpe.decode(string)
        print()
        print(f"result: {detok_str}")
        
        with open('./result.txt', mode='a') as of:
            of.write(f'{detok_str}\n')
        
        scorer.add_string(detok_str, str(text[0]))
    
    print(scorer.result_string())

if __name__ == "__main__":
    try:
        from .bpe import GPT2BPE
    except ImportError:
        from bpe import GPT2BPE
    
    from fairseq import utils
    
    def set_logger():
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(filename='./model.log', encoding='utf-8', mode='w')
        formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    logger = set_logger()
    
        
    dict_path_or_url = './IAM/gpt2_with_mask.dict.txt'
    if dict_path_or_url is not None and dict_path_or_url.startswith('https') :
        content = urllib.request.urlopen(dict_path_or_url).read().decode()
        dict_file = io.StringIO(content)
        tgt_dict = Dictionary.load(dict_file)
    elif os.path.exists(dict_path_or_url):
        content = io.StringIO(dict_path_or_url)
        tgt_dict = Dictionary.load(dict_path_or_url)
    else:
        raise ValueError('Could not find tgt_dictionary.')
    
    root_dir = '/kaggle/input/iam-handwriting/IAM/image/'
    df = pd.read_fwf('/kaggle/input/iam-handwriting/IAM/gt_test.txt', header=None)
    df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
    del df[2]
    df['file_name'] = df['file_name'].apply(lambda x: x + 'g' if x.endswith('jp') else x)
    
    dataset = STRDataset(root_dir=root_dir, df=df)
    data = DataLoader(dataset=dataset, batch_size=1)
    
    state = torch.load('/kaggle/input/trocr-base-model/trocr-base.pt')
    main(state, data)