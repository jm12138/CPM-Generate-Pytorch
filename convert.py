import os
import torch
import numpy as np

m0 = torch.load('./model-v1/80000/mp_rank_00_model_states.pt', map_location='cpu')
m1 = torch.load('./model-v1/80000/mp_rank_01_model_states.pt', map_location='cpu')

if not os.path.exists('numpy'):
    os.mkdir('numpy')
for x, y in zip(m0['module'].items(),  m1['module'].items()):
    n0, p0 = x
    n1, p1 = y
    if not (p0.numpy()==p1.numpy()).all():
        if 'attention.query_key_value.weight' in n0:
            w1 = torch.cat([p0[:1280, :], p1[:1280, :]], dim=0).transpose(1, 0)
            w2 = torch.cat([p0[1280:1280*2, :], p1[1280:1280*2, :]], dim=0).transpose(1, 0)
            w3 = torch.cat([p0[1280*2:, :], p1[1280*2:, :]], dim=0).transpose(1, 0)
            p = torch.cat([w1, w2, w3], dim=1).transpose(1, 0)
        elif 'attention.query_key_value.bias' in n0:
            w1 = torch.cat([p0[:1280], p1[:1280]], dim=0)
            w2 = torch.cat([p0[1280:1280*2], p1[1280:1280*2]], dim=0)
            w3 = torch.cat([p0[1280*2:], p1[1280*2:]], dim=0)
            p = torch.cat([w1, w2, w3], dim=0)
        elif 'attention.dense.weight' in n0:
            p = torch.cat([p0, p1], dim=1)
        elif 'mlp.dense_h_to_4h.weight' in n0:
            p = torch.cat([p0, p1], dim=0)
        elif 'mlp.dense_h_to_4h.bias' in n0:
            p = torch.cat([p0, p1], dim=0)
        elif 'mlp.dense_4h_to_h.weight' in n0:
            p = torch.cat([p0, p1], dim=1)
        elif 'word_embeddings' in n0:
            p = torch.cat([p0, p1], dim=0)
        else: 
            print('other')
            print(n0, p0.numpy().shape)
            print(n1, p1.numpy().shape)
    else:
        p =  p0
    m0['module'][n0]=p

torch.save(m0['module'], 'save.pth')