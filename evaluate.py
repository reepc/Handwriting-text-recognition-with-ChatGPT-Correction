from fairseq import utils
from torch.utils.data import DataLoader
import torch
import pandas as pd

from tqdm.auto import tqdm 

try:
    from .model import create_TrOCR_model
    from .bpe import GPT2BPE
    from .data_preprocess import STRDataset
    from .score import CERScorer
except ImportError:
    from model import create_TrOCR_model
    from bpe import GPT2BPE
    from data_preprocess import STRDataset
    from score import CERScorer

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%'
)

logger = logging.getLogger('Evaluating')

def main(state, data):
    bpe = GPT2BPE()
    model = create_TrOCR_model(state)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.to(device)
    model.eval()
    
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
        print(detok_str)
        
        with open('./result.txt', mode='a') as of:
            of.write(detok_str)
        
        scorer.add_string(detok_str, str(text[0]))
    
    print(scorer.result_string())

if __name__ == "__main__":
    root_dir = '/kaggle/input/iam-handwriting/IAM/image/'
    df = pd.read_fwf('/kaggle/input/iam-handwriting/IAM/gt_test.txt', header=None)
    df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
    del df[2]
    df['file_name'] = df['file_name'].apply(lambda x: x + 'g' if x.endswith('jp') else x)
    
    dataset = STRDataset(root_dir=root_dir, df=df)
    data = DataLoader(dataset=dataset, batch_size=1)
    
    state = torch.load('/kaggle/input/trocr-base-model/trocr-base.pt')
    main(state, data)