import torch.nn as nn

# TODO
class TransformerAdapter(nn.Module):
    """
    A transformer adapter, using torch.nn.Module.
    """
    def __init__(self, d_model: int =768):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model = d_model,
            num_encoder_layers=4,
            num_decoder_layers=4,
            activation='gelu'
        )
    
    def forward(self, src, prev):
        self.transformer(src)
        
class TransformerEncoderAdapter(nn.Module):
    """
    A Transformer encoder adapter, using torch.nn.Module.
    """
    def __init__(self, d_model=768):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, src):
        out = self.encoder(src)
        out = self.fc(out)
        
        return out

class ModalAdapter(nn.Module):
    """
    Modify from E2TIMT (https://arxiv.org/pdf/2305.05166.pdf)
    """
    
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, 8)
        self.norm = nn.LayerNorm((768,), eps=1e-06)
        self.ff = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        out = self.attn(x, x, x)
        out = self.norm(out)
        out += x
        temp = out
        
        out = self.ff(out)
        out = self.norm(out)
        out += temp
        return out

if __name__ == '__main__':
    pass