import torch

ost = torch.load('./trocr-base.pt')['model']
encoder_state = {}
decoder_state = {}

for key, value in ost.items():
    new_key = key.replace('encoder.','').replace('decoder.','')
    if key.startswith('encoder.'):
        encoder_state[new_key] = value
    else:
        decoder_state[new_key] = value

torch.save(encoder_state, 'encoder.pt')
torch.save(decoder_state, 'decoder.pt')