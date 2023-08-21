from fairseq.models import FairseqEncoder, FairseqEncoderDecoderModel
from fairseq.models.transformer import TransformerDecoder, Embedding, TransformerModel
from fairseq.data import Dictionary

from timm.models.factory import create_model
from register_model import *

import torch
import torch.nn as nn

import argparse, logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%',
    handlers = [logging.FileHandler('my.log', 'w', 'utf-8'),]
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
        output = self.encoder(input)
        if self.adapter:
            self.adapter(output)
        

def create_TrOCR_model(state):
    cfg = state['cfg']
    
    encoder = Encoder(args=cfg['model'], dictionary=None)
    logger.info(f"Encoder:\n{encoder}")
    
    decoder = Decoder(args=cfg['model']).build_decoder()
    logger.info(f"Decoder:\n{decoder}")
    
    TrOCR = TrOCRModel(encoder=encoder, decoder=decoder)


if __name__ == '__main__':
    state = torch.load('./trocr-base.pt')
    create_TrOCR_model(state)