from fairseq.data.encoders.gpt2_bpe_utils import get_encoder
from fairseq import file_utils

class GPT2BPE(object):
    """To process output tokens to words, or process words to tokens"""
    
    def __init__(self):
        encoder_json = file_utils.cached_path('kaggle/working/HTRCC/bpe/gpt2_bpe_encoder.json')
        vocab_bpe = file_utils.cached_path('kaggle/working/HTRCC/bpe/gpt2_vocab.bpe')
        self.bpe = get_encoder(encoder_json, vocab_bpe)
    
    def encode(self, x: str) -> str:
        return " ".join(map(str, self.bpe.encode(x)))
    
    def decode(self, x :str) -> str:
        return self.bpe.decode(
            [int(tok) if tok not in {"<unk>", "<mask>"} else tok for tok in x.split()]
        )
    
    def is_beginning(self, x: str) -> bool:
        return self.decode(x).startswith(" ")