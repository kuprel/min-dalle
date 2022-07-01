import os
import numpy
from typing import Dict
from flax.traverse_util import flatten_dict
from flax.serialization import msgpack_restore
import torch
torch.set_grad_enabled(False)


def load_vqgan_torch_params(path: str) -> Dict[str, torch.Tensor]:
    with open(os.path.join(path, 'flax_model.msgpack'), "rb") as f:
        params: Dict[str, numpy.ndarray] = msgpack_restore(f.read())

    P: Dict[str, numpy.ndarray] = flatten_dict(params, sep='.')

    for i in list(P.keys()):
        j = i
        if 'up' in i or 'down' in i:
            j = i.replace('_', '.')
            j = j.replace('proj.out', 'proj_out')
            j = j.replace('nin.short', 'nin_short')
        if 'bias' in i:
            P[j] = P.pop(i)
        elif 'scale' in i:
            j = j.replace('scale', 'weight')
            P[j] = P.pop(i)
        elif 'kernel' in i:
            j = j.replace('kernel', 'weight')
            P[j] = P.pop(i).transpose(3, 2, 0, 1)

    for i in P:
        P[i] = torch.tensor(P[i])

    P['embedding.weight'] = P.pop('quantize.embedding.embedding')

    for i in list(P):
        if i.split('.')[0] in ['encoder', 'quant_conv']:
            P.pop(i)
    
    return P


def load_dalle_bart_flax_params(path: str) -> Dict[str, numpy.ndarray]:
    with open(os.path.join(path, "flax_model.msgpack"), "rb") as f:
        params = msgpack_restore(f.read())

    for codec in ['encoder', 'decoder']:
        k = 'FlaxBart{}Layers'.format(codec.title())
        P: dict = params['model'][codec]['layers'][k]
        P['pre_self_attn_layer_norm'] = P.pop('LayerNorm_0')
        P['self_attn_layer_norm'] = P.pop('LayerNorm_1')
        P['self_attn'] = P.pop('FlaxBartAttention_0')
        if codec == 'decoder':
            P['pre_encoder_attn_layer_norm'] = P.pop('LayerNorm_2')
            P['encoder_attn_layer_norm'] = P.pop('LayerNorm_3')
            P['encoder_attn'] = P.pop('FlaxBartAttention_1')
        P['glu']: dict = P.pop('GLU_0')
        P['glu']['ln0'] = P['glu'].pop('LayerNorm_0')
        P['glu']['ln1'] = P['glu'].pop('LayerNorm_1')
        P['glu']['fc0'] = P['glu'].pop('Dense_0')
        P['glu']['fc1'] = P['glu'].pop('Dense_1')
        P['glu']['fc2'] = P['glu'].pop('Dense_2')

    for codec in ['encoder', 'decoder']:
        layers_params = params['model'][codec].pop('layers')
        params['model'][codec] = {
            **params['model'][codec], 
            **layers_params
        }
    
    model_params = params.pop('model')
    params = {**params, **model_params}

    params['decoder']['lm_head'] = params.pop('lm_head')

    return params


def convert_dalle_bart_torch_from_flax_params(
    params: dict,
    layer_count: int,
    is_encoder: bool
) -> dict:
    P: Dict[str, numpy.ndarray] = flatten_dict(params, sep='.')

    for i in P:
        P[i] = torch.tensor(P[i]).to(torch.float16)

    for i in list(P):
        if 'kernel' in i:
            j = i.replace('kernel', 'weight')
            P[j] = P.pop(i).transpose(-1, -2)
        elif 'scale' in i:
            j = i.replace('scale', 'weight')
            P[j] = P.pop(i)

    for i in list(P):
        j = 'FlaxBart{}Layers'.format('Encoder' if is_encoder else 'Decoder')
        if j in i:
            for l in range(layer_count):
                k = i.replace(j, 'layers.' + str(l))
                P[k] = P[i][l]
            P.pop(i)

    P['embed_tokens.weight'] = P.pop('embed_tokens.embedding')
    P['embed_positions.weight'] = P.pop('embed_positions.embedding')
    return P


def convert_and_save_torch_params(is_mega: bool, model_path: str):
    print("converting params to torch")
    layer_count = 24 if is_mega else 12
    flax_params = load_dalle_bart_flax_params(model_path)
    encoder_params = convert_dalle_bart_torch_from_flax_params(
        flax_params['encoder'],
        layer_count=layer_count,
        is_encoder=True
    )
    decoder_params = convert_dalle_bart_torch_from_flax_params(
        flax_params['decoder'],
        layer_count=layer_count,
        is_encoder=False
    )

    for i in decoder_params:
        decoder_params[i] = decoder_params[i].to(torch.float16)
    
    for i in encoder_params:
        encoder_params[i] = encoder_params[i].to(torch.float16)

    detoker_params = load_vqgan_torch_params('./pretrained/vqgan')
    detoker_path = os.path.join('pretrained', 'vqgan', 'detoker.pt')

    torch.save(encoder_params, os.path.join(model_path, 'encoder.pt'))
    torch.save(decoder_params, os.path.join(model_path, 'decoder.pt'))
    torch.save(detoker_params, detoker_path)