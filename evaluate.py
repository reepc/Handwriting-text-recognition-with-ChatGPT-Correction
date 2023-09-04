from fairseq import utils
from torch.utils.data import DataLoader
import torch

from tqdm.auto import tqdm 

try:
    from .model import create_TrOCR_model
    from .bpe import GPT2BPE
    from .data_preprocess import STRDataset
except ImportError:
    from model import create_TrOCR_model
    from bpe import GPT2BPE
    from data_preprocess import STRDataset
    
def main(state, img_path):
    bpe = GPT2BPE()
    model = create_TrOCR_model(state)
    dataset = STRDataset(img_path)
    data = DataLoader(dataset=dataset, batch_size=1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.to(device)
    model.eval()
    
    for img in tqdm(data):
        decoder_out = model.forward(img)
        
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
    
    return detok_str

if __name__ == "__main__":
    img_path = '/kaggle/input/iam-handwriting/IAM/image/c04-110-00.jpg'
    state = torch.load('/kaggle/input/trocr-base-model/trocr-base.pt')
    print(main(state, img_path))