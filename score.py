from fairseq.scoring import BaseScorer, register_scorer
from fairseq.dataclass import FairseqDataclass
import fastwer

from dataclasses import dataclass

@dataclass
class CERScorerConfig(FairseqDataclass):
    name: str = 'default'

@register_scorer("cer", dataclass=FairseqDataclass)
class CERScorer(BaseScorer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.refs = []
        self.preds = []

    def add_string(self, ref, pred):
        self.refs.append(ref)
        self.preds.append(pred)
    
    def score(self):
        return fastwer.score(self.preds, self.refs, char_level=True)

    def result_string(self) -> str:
        return f"CER: {self.score():.2f}"