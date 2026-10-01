import os 
import sys 
import pickle 

import torch
import numpy as np 

from model import Model

data_dir = './data/'
# model cfg 
embed_n = 0
embed_d = 0
block_size = 0
pos_embed_d = 0   
hidden_d = 0
win_d = 0 
# training 
epochs = ... 
steps = ... 

def _handle_args(): 
    for arg in sys.argv[1:]: 
        if arg.endswith('.py'):
            print(f'overriding config with: {arg}')
            with open(arg) as f: print(f.read())
            exec(open(arg).read(), globals())

def _get_meta() -> dict:
    return pickle.load(open(os.path.join(data_dir, 'meta.pkl'), 'rb'))

def _get_sample(split: str):
    # randomly sample pointer and randomly sample size 
    data = np.memmap(os.path.join(data_dir, split + '.bin'), dtype=np.uint16, mode='r')
    pointer = torch.randint(len(data) - block_size, (1,))
    size = torch.randint(1,block_size+1, (1,))
    x = torch.from_numpy(data[pointer:pointer+size].astype(np.int64)).unsqueeze(0) # [1, size]
    y = torch.tensor(data[pointer+size], dtype=torch.long).unsqueeze(0) # [1, 1]
    return x, y 
    
def encode(stoi: dict, s: str): 
    return [stoi[ch] for ch in s]

def decode(itos: dict, ids: list[int]):
    return ''.join(itos[id_] for id_ in ids)

if __name__ == '__main__': 
    _handle_args()
    meta = _get_meta()
    stoi, itos = meta['stoi'], meta['itos'] 
    model = Model(embed_n, embed_d, block_size, pos_embed_d, hidden_d, win_d)

    x, y = _get_sample('val')
    logits, loss = model(x, y)
    print(logits.shape)
    print(loss.item())
