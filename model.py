from fairseq.models import FairseqEncoder, FairseqEncoderDecoderModel
from fairseq.models.transformer import TransformerDecoder, Embedding
from fairseq.data import Dictionary
from fairseq.search import BeamSearch

from timm.models.factory import create_model
from register_model import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import Tensor

import argparse, logging, math, sys, os, io
import urllib.request
from typing import List, Dict, Optional

from tqdm.auto import tqdm

try:
    from .dataset import STRDataset
    from .adapter import *
except ImportError:
    from dataset import STRDataset
    from adapter import *

def set_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(filename='./model.log', encoding='utf-8', mode='w')
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

logger = set_logger()
class Encoder(FairseqEncoder):
    def __init__(self, args, dictionary: Dictionary):
        super().__init__(dictionary)
        logger.debug('Building Encoder...')
        
        if hasattr(args, 'only_keep_pretrained_encoder_structure') and args.only_keep_pretrained_encoder_structure:
            pretrained = False
        else:
            pretrained = True

        if 'custom_size' in args.deit_arch:
            self.deit = create_model(args.deit_arch, pretrained=pretrained, img_size=args.input_size, ape=args.ape, mask_ratio=args.mask_ratio)
        else:
            self.deit = create_model(args.deit_arch, pretrained=pretrained, ape=args.ape, mask_ratio=args.mask_ratio)

        logger.debug("Encoder builded")
    
    def forward(self, img):
        x, encoder_embedding = self.deit.forward_features(img)  # bs, n + 2, dim
        x = x.transpose(0, 1) # n + 2, bs, dim

        encoder_padding_mask = torch.zeros(*x.shape[:2]).transpose(0, 1).to(img.device)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        _encoder_out = encoder_out['encoder_out'][0]
        _encoder_padding_mask = encoder_out['encoder_padding_mask'][0]
        _encoder_embedding = encoder_out['encoder_embedding'][0]

        return {
            "encoder_out": [_encoder_out.index_select(1, new_order)],
            "encoder_padding_mask": [_encoder_padding_mask.index_select(0, new_order)],  # B x T
            "encoder_embedding": [_encoder_padding_mask.index_select(0, new_order)],  # B x T x C
            "encoder_states": [], 
            "src_tokens": [],
            "src_lengths": [],
        }

class Decoder:
    def __init__(self, args, tgt_dictionary: Dictionary):
        self.args = args
        self.tgt_dict = tgt_dictionary
        
    
    def read_roberta_args(self, roberta_model: argparse.Namespace):
        args = argparse.Namespace(**vars(roberta_model))
        attr_map = [
            ("encoder_attention_heads", "decoder_attention_heads"),
            ("encoder_embed_dim", "decoder_embed_dim"),
            ("encoder_embed_dim", "decoder_output_dim"),
            ("encoder_normalize_before", "decoder_normalize_before"),
            ("encoder_layers_to_keep", "decoder_layers_to_keep"),
            ("encoder_ffn_embed_dim", "decoder_ffn_embed_dim"),
            ("encoder_layerdrop", "decoder_layerdrop"),
            ("encoder_layers", "decoder_layers"),
            ("encoder_learned_pos", "decoder_learned_pos"),
            ("max_positions", "max_target_positions"),
        ]
        for k1, k2 in attr_map:
            setattr(args, k2, getattr(roberta_model, k1))

        args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
        args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
        args.share_decoder_input_output_embed = not roberta_model.untie_weights_roberta
        return args
    
    def build_decoder(self):
        logger.debug("Building decoder...")
        
        roberta = torch.hub.load('pytorch/fairseq:main', 'roberta.large')
        roberta.model.args.encoder_layers = 12
        roberta.model.args.fp16 = self.args.fp16
        
        roberta_args = self.read_roberta_args(roberta.model.args)
        roberta_args.encoder_embed_dim = self.args.encoder_embed_dim

        decoder = TransformerDecoder(
            roberta_args,
            None,
            self.build_embedding(),
            no_encoder_attn=False
        )
        
        logger.debug("Decoder builded")
        return decoder
    
    def build_embedding(self):
        logger.debug("Building embedding...")
        embed_dim = self.args.decoder_embed_dim
        
        num_embeddings = len(self.tgt_dict)
        padding = self.tgt_dict.pad()

        embeddings = Embedding(num_embeddings=num_embeddings, embedding_dim=embed_dim, padding_idx=padding)
        
        logger.debug("Embedding builded")
        return embeddings

class TrOCRModel(FairseqEncoderDecoderModel):
    def __init__(
        self,
        encoder: FairseqEncoder,
        decoder: TransformerDecoder,
        adapter: nn.Module = None,
        tgt_dictionary: Dictionary = None,
        beam_size: int = 5,
        max_len_a: float = 0.0,
        max_len_b: int = 200,
        min_len: int = 1
        
    ):
        super().__init__(encoder, decoder)
        self.tgt_dictionary = tgt_dictionary
        self.pad = self.tgt_dictionary.pad()
        self.eos = self.tgt_dictionary.eos()
        self.unk = self.tgt_dictionary.unk()
        self.vocab_size = len(self.tgt_dictionary)
        
        self.unk_penalty = 0.0
        self.encoder = encoder
        self.decoder = decoder
        self.adapter = adapter
        
        self.beam_size = beam_size
        self.search = BeamSearch(tgt_dictionary)
        
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.temperature = 1.0
    
    def forward(self, input):
        encoder_out = self.encoder(input)

        if self.adapter is not None:
            if hasattr(self.adapter, 'encoder') or hasattr(self.adapter, 'attn'):
                encoder_out['encoder_out'][0] = self.adapter(encoder_out['encoder_out'][0])
            elif hasattr(self.adapter, 'transformer'):
                for i in range(10):
                    pass
        
        src_length = encoder_out['encoder_padding_mask'][0].eq(0).long().sum(dim=1)
        src_tokens = encoder_out['encoder_padding_mask'][0]
        
        bsz, src_len = src_tokens.size()[:2]
        
        self.search.init_constraints(None, self.beam_size)
        
        max_len = min(self.max_decoder_positions() - 1, int(self.max_len_a * src_len + self.max_len_b))
        
        new_order = (torch.arange(bsz).view(-1, 1).repeat(1, self.beam_size).view(-1)).to(src_tokens.device).long()
        encoder_out = self.encoder.reorder_encoder_out(encoder_out, new_order)
        # encoder_out should be a list
        assert encoder_out is not None
        
        scores = torch.zeros(bsz * self.beam_size, max_len + 1).to(src_tokens).float()
        tokens = torch.zeros(bsz * self.beam_size, max_len + 2).to(src_tokens).long().fill_(self.pad)
        tokens[:, 0] = self.eos
        
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})]
        )
        
        cands_to_ignore = torch.zeros(bsz, self.beam_size).to(src_tokens).eq(-1)
        
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)]
        )
        
        finished = [False for i in range(bsz)]
        remainings = bsz
        cand_size = 2 * self.beam_size
        bbsz_offsets = (torch.arange(0, bsz) * self.beam_size).unsqueeze(1).type_as(tokens).to(src_tokens.device)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(src_tokens.device)
        
        attn: Optional[Tensor] = None
        reorder_states: Optional[Tensor] = None
        batch_idx: Optional[Tensor] = None
        
        original_batch_idxs = torch.arange(0, bsz).type_as(tokens)
        
        for step in range(max_len + 1):
            if reorder_states is not None:
                if batch_idx is not None:
                    corr = batch_idx - torch.arange(batch_idx.numel()).type_as(batch_idx)
                    reorder_states.view(-1, self.beam_size).add_(corr.unsqueeze(-1) * self.beam_size)
                    original_batch_idx = original_batch_idx[batch_idx]

                self.decoder.reorder_incremental_state(incremental_states, reorder_states)
                encoder_out = self.encoder.reorder_encoder_out(encoder_out, reorder_states)
            
            probs, attn_scores = self.decoder_forward(
                tokens[:, : step + 1],
                encoder_out,
                incremental_states,
                self.temperature
            )
            
            probs[probs != probs] = torch.tensor(-math.inf).to(probs)
            
            probs[:, self.pad] = -math.inf
            probs[:, self.unk] -= self.unk_penalty
            
            if step >= max_len:
                probs[:, : self.eos] = -math.inf
                probs[:, self.eos+1 :] = -math.inf
                
            if step < self.min_len:
                probs[:, self.eos] = -math.inf
                
            if attn_scores is not None:
                if attn is None:
                    attn = torch.empty(bsz * self.beam_size, attn_scores.size(1), max_len + 2).to(scores)
                
                attn[:, :, step + 1].copy_(attn_scores)
            
            scores = scores.type_as(probs)
            eos_bbsz_idx = torch.empty(0).to(tokens)
            eos_scores = torch.empty(0).to(scores)
            
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                probs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, self.beam_size, -1)[:, :, :step],
                tokens[:, : step + 1],
                original_batch_idxs
            )
            
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)
            
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :self.beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)
            
            eos_bbsz_idx = torch.masked_select(cand_bbsz_idx[:, :self.beam_size], eos_mask[:, :self.beam_size])
            
            finals: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(cand_scores[:, :self.beam_size], mask=eos_mask[:, :self.beam_size])
                
                finals = self.final_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    self.beam_size,
                    attn,
                    src_length,
                    max_len
                )
                remainings -= len(finals)
            
            assert remainings >= 0
            if remainings == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f'{step} < {max_len}'
            
            if len(finals) > 0:
                new_bsz = bsz - len(finals)
                
                batch_mask = torch.ones(bsz, dtype=torch.bool, device=cand_indices.device)
                batch_mask[finals] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                src_length = src_length[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * self.beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * self.beam_size, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * self.beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None
            
            eos_mask[:, :self.beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :self.beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)]
            )
            
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=self.beam_size, dim=1, largest=False
            )
            
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :self.beam_size]
            assert (~cands_to_ignore).any(dim=1).all()
            
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)
            
            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)
            
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            tokens.view(bsz, self.beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, self.beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_states = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
            
        return finalized
            
    
    def final_hypos(
        self,
        step: int,
        bbsz_idx,
        eos_scores,
        tokens,
        scores,
        finalized: List[List[Dict[str, Tensor]]],
        finished: List[bool],
        beam_size: int,
        attn: Optional[Tensor],
        src_lengths,
        max_len: int,
    ):
        assert bbsz_idx.numel() == eos_scores.numel()
        tokens_clone = tokens.index_select(0, bbsz_idx)[
            :, 1 : step + 2
        ]  # skip the first index, which is EOS

        tokens_clone[:, step] = self.eos
        attn_clone = (
            attn.index_select(0, bbsz_idx)[:, :, 1 : step + 2]
            if attn is not None
            else None
        )

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
        pos_scores[:, step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]
        
        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)
        cum_fin_tensor = torch.tensor(cum_unfin, dtype=torch.int).to(bbsz_idx)

        unfin_idx = torch.div(bbsz_idx, beam_size, rounding_mode="trunc")
        sent = unfin_idx + torch.index_select(cum_fin_tensor, 0, unfin_idx)

        # Create a set of "{sent}{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        # For every finished beam item
        # sentence index in the current (possibly reduced) batch
        seen = (sent << 32) + unfin_idx
        unique_seen: List[int] = torch.unique(seen).tolist()

        sent_list: List[int] = sent.tolist()
        for i in range(bbsz_idx.size()[0]):
            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent_list[i]]) < beam_size:
                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)

                finalized[sent_list[i]].append(
                    {
                        "tokens": tokens_clone[i],
                        "score": eos_scores[i],
                        "attention": hypo_attn,  # src_len x tgt_len
                        "alignment": torch.empty(0),
                        "positional_scores": pos_scores[i],
                    }
                )

        newly_finished: List[int] = []
        for unique_s in unique_seen:
            # check termination conditions for this sentence
            unique_sent: int = unique_s >> 32
            unique_unfin_idx: int = unique_s - (unique_sent << 32)

            if not finished[unique_sent] and self.is_finished(
                step, max_len, len(finalized[unique_sent]),
            ):
                finished[unique_sent] = True
                newly_finished.append(unique_unfin_idx)

        return newly_finished
    
    def is_finished(
        self,
        step,
        max_len,
        finalized_sent_len
    ):
        assert finalized_sent_len <= self.beam_size
        if finalized_sent_len == self.beam_size or step == max_len:
            return True
        return False
        
    def decoder_forward(
        self,
        tokens,
        encoder_out,
        incremental_state,
        temperature: float = 1.0
    ):
        probs = []
        
        decoder_out = self.decoder(tokens, encoder_out, incremental_state = incremental_state[0])
        attn = decoder_out[1]['attn']
        attn = attn[0][:, -1, :]
        
        if len(decoder_out) > 1 and decoder_out[1] is not None:
            if isinstance(decoder_out[1], Tensor):
                attn = decoder_out[1]
            else:
                attn_holder = decoder_out[1]["attn"]
                if isinstance(attn_holder, Tensor):
                    attn = attn_holder
                elif attn_holder is not None:
                    attn = attn_holder[0]
            if attn is not None:
                attn = attn[:, -1, :]
        
        decoder_tuple_out = (
            decoder_out[0][:, -1:, :].div_(temperature),
            None if len(decoder_out) <= 1 else decoder_out[1]
        )
        probs = self.decoder.get_normalized_probs(
            decoder_tuple_out,
            log_probs=True
        )
        probs = probs[:, -1, :]
        
        return probs, attn
    
    def max_decoder_positions(self):
        return min([self.encoder.max_positions()] + [sys.maxsize])
        
        
def create_TrOCR_model(model_args, tgt_dict):
    encoder = Encoder(args=model_args, dictionary=None)
    logger.debug(f"Encoder:\n{encoder}")
    encoder.load_state_dict(torch.load(f'{os.getcwd()}/encoder.pt'))
    
    decoder = Decoder(args=model_args, tgt_dictionary=tgt_dict).build_decoder()
    logger.debug(f"Decoder:\n{decoder}")
    decoder.load_state_dict(torch.load(f'{os.getcwd()}/decoder.pt'))
    
    TrOCR = TrOCRModel(encoder=encoder, decoder=decoder, tgt_dictionary=tgt_dict)
    
    return TrOCR

if __name__ == '__main__':
    try:
        from .bpe import GPT2BPE
    except ImportError:
        from bpe import GPT2BPE
    
    from fairseq import utils
        
    dict_path_or_url = './gpt2_with_mask.dict.txt'
    if dict_path_or_url is not None and dict_path_or_url.startswith('https') :
        content = urllib.request.urlopen(dict_path_or_url).read().decode()
        dict_file = io.StringIO(content)
        tgt_dict = Dictionary.load(dict_file)
        
    elif os.path.exists(dict_path_or_url):
        content = io.StringIO(dict_path_or_url)
        tgt_dict = Dictionary.load(dict_path_or_url)
        
    else:
        raise ValueError('Could not find tgt_dictionary.')

    model_args = torch.load('./state.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_set = STRDataset()
    data = DataLoader(dataset=data_set, batch_size=1, shuffle=False)
    
    model = create_TrOCR_model(model_args, tgt_dict)
    model.eval().to(device)
    
    bpe = GPT2BPE()
    with torch.no_grad():
        for img in tqdm(data):
            outs = model(img.to(device))
    
            outs = outs[0][0]
            
            tokens, strs, alignment = utils.post_process_prediction(
                hypo_tokens=outs['tokens'].int().cpu(),
                src_str='',
                alignment=outs['alignment'],
                align_dict=None,
                tgt_dict=model.tgt_dictionary,
                remove_bpe=None,
                extra_symbols_to_ignore={2}
            )
            
            logger.info(bpe.decode(strs))