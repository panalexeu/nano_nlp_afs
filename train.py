import sys 
import pickle 

import torch

from model import Model

embed_n = 0
embed_d = 0
hidden_d = 0
win_d = 0 

def _handle_args(): 
    for arg in sys.argv[1:]: 
        if arg.endswith('.py'):
            print(f'overriding config with: {arg}')
            with open(arg) as f: print(f.read())
            exec(open(arg).read(), globals())

def _get_meta() -> dict:
    path = './meta.pkl' 
    return pickle.load(open(path, 'rb'))

def encode(stoi: dict, s: str): 
    return [stoi[ch] for ch in s]

def decode(itos: dict, ids: list[int]):
    return ''.join(itos[id_] for id_ in ids)

if __name__ == '__main__': 
    _handle_args()
    meta = _get_meta()
    stoi, itos = meta['stoi'], meta['itos'] 
    model = Model(embed_n, embed_d, hidden_d, win_d)

    s = 'hello world!'
    ids = torch.tensor(encode(stoi, s), dtype=torch.long).unsqueeze(0) # add batch dim?
    res = model(ids)
    print(res.shape)
