import torch
import os

def process_model(model_path="./trocr-base.pt"):
    ost = torch.load(model_path)['model']
    encoder_state = {}
    decoder_state = {}

    for key, value in ost.items():
        new_key = key.replace('encoder.','').replace('decoder.','')
        if key.startswith('encoder.'):
            encoder_state[new_key] = value
        else:
            decoder_state[new_key] = value

    torch.save(encoder_state, './encoder.pt')
    torch.save(decoder_state, './decoder.pt')

    torch.save(torch.load(model_path)['cfg']['model'], './state.pt')
    os.remove(model_path)