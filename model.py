from fairseq.models import FairseqEncoder, FairseqEncoderDecoderModel
from fairseq.models.transformer import TransformerDecoder, Embedding
from fairseq.data import Dictionary
from fairseq.search import BeamSearch

from timm.models.factory import create_model
from register_model import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import argparse, logging
from tqdm.auto import tqdm

from data_preprocess import STR

from typing import List, Dict, Optional

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%'
)

logger = logging.getLogger('Model Creating')

class Encoder(FairseqEncoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        logger.info('Building Encoder...')
        
        if hasattr(args, 'only_keep_pretrained_encoder_structure') and args.only_keep_pretrained_encoder_structure:
            pretrained = False
        else:
            pretrained = True

        if 'custom_size' in args.deit_arch:
            self.deit = create_model(args.deit_arch, pretrained=pretrained, img_size=args.input_size, ape=args.ape, mask_ratio=args.mask_ratio)
        else:
            self.deit = create_model(args.deit_arch, pretrained=pretrained, ape=args.ape, mask_ratio=args.mask_ratio)

        self.fp16 = args.fp16
        logger.info("Encoder builded")
        
    def forward(self, img):
        if self.fp16:
            imgs = img.half()
        
        x, encoder_embedding = self.deit.forward_features(imgs)  # bs, n + 2, dim
        x = x.transpose(0, 1) # n + 2, bs, dim

        encoder_padding_mask = torch.zeros(*x.shape[:2]).transpose(0, 1).to(imgs.device)

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
    def __init__(self, args):
        self.args = args
    
    def read_roberta_args(self, roberta_model):
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
            # should this be set from here ?
            ("max_positions", "max_target_positions"),
        ]
        for k1, k2 in attr_map:
            setattr(args, k2, getattr(roberta_model, k1))

        args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
        args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
        args.share_decoder_input_output_embed = not roberta_model.untie_weights_roberta
        return args
    
    def build_decoder(self):
        logger.info("Building decoder...")
        
        roberta = torch.hub.load('pytorch/fairseq:main', 'roberta.large')
        roberta.model.args.encoder_layers = 12
        roberta.model.fp16 = self.args.fp16
        
        roberta_args = self.read_roberta_args(roberta.model.args)
        roberta_args.encoder_embed_dim = self.args.encoder_embed_dim

        decoder = TransformerDecoder(
            roberta_args,
            None,
            self.build_embedding(),
            no_encoder_attn=False
        )
        
        logger.info("Decoder builded")
        return decoder
    
    def build_embedding(self):
        logger.info("Building embedding...")
        embed_dim = self.args.decoder_embed_dim
        dictionary = Dictionary.load('./IAM/gpt2.dict.txt')
        
        num_embeddings = len(dictionary)
        padding = dictionary.pad()

        embeddings = Embedding(num_embeddings=num_embeddings, embedding_dim=embed_dim, padding_idx=padding)
        
        logger.info("Embedding builded")
        return embeddings

class Adapter:
    pass


class TrOCRModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder, adapter=None):
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
        self.adapter = adapter
    
    def forward(self, input):
        encoder_out = self.encoder.forward(input)
        
        src_length = encoder_out['encoder_padding_mask'][0].eq(0).long()
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
            List[Dict[str, Dict[str, Optional[torch.Tensor]]]],
            [torch.jit.annotate(Dict[str, Dict[str, Optional[torch.Tensor]]], {})]
        )
        
        cands_to_ignore = torch.zeros(bsz, self.beam_size).to(src_tokens).eq(-1)
        
        sentences = torch.jit.annotate(
            List[List[Dict[str, torch.Tensor]]],
            [torch.jit.annotate(List[Dict[str, torch.Tensor]], []) for i in range(bsz)]
        )
        
        finished = [False for i in range(bsz)]
        bbsz = (torch.arange(0, bsz) * self.beam_size).unsqueeze(1).type_as(tokens).to(src_tokens.device)
        
        reorder_states: Optional[torch.Tensor] = None
        batch_idx = Optional[torch.Tensor] = None
        
        for step in range(max_len + 1):
            
            if reorder_states is not None:
                if batch_idx is not None:
                    corr = batch_idx - torch.arange(batch_idx.numel()).type_as(batch_idx)
                    reorder_states.view(-1, self.beam_size).add_(corr.unsqueeze(-1) * self.beam_size)
                    original_batch_idx = original_batch_idx[batch_idx]

                self.decoder.reorder_incremental_state(incremental_states, reorder_states)
                encoder_out = self.encoder.reorder_encoder_out(encoder_out, reorder_states)
            
            if incremental_states is not None:
                decoder_out = self.decoder.forward(tokens[:, : step + 1], encoder_out, incremental_states)
            else:
                decoder_out = self.decoder.forward(tokens[:, : step + 1], encoder_out)
        

def create_TrOCR_model(state):
    cfg = state['cfg']
    
    encoder = Encoder(args=cfg['model'], dictionary=None)
    logger.info(f"Encoder:\n{encoder}")
    encoder.load_state_dict(torch.load('./encoder.pt'))
    
    decoder = Decoder(args=cfg['model']).build_decoder()
    logger.info(f"Decoder:\n{decoder}")
    decoder.load_state_dict(torch.load('./decoder.pt'))
    
    TrOCR = TrOCRModel(encoder=encoder, decoder=decoder)
    
    return TrOCR

def test(state, device, image):
    model = create_TrOCR_model(state).to(device)
    model.eval()

    for img in tqdm(image):
        result = model(img.to(device))
    return result


if __name__ == '__main__':
    state = torch.load('./trocr-base.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_set = STR('./IAM/image/c04-110-00.jpg')
    data = DataLoader(dataset=data_set, batch_size=1)
    print(test(state, device, data))