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
epochs = 0 
steps = 0
eval_steps = 0 
lr = 0 
logging_steps = 1000

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

def _save_model(model: torch.nn.Module, ckpt_path: str = './ckpt.pt'):
    dict_ = {
        'cfg': {
            'embed_n': model.embed_n,
            'embed_d': model.embed_d, 
            'pos_embed_n': model.pos_embed_n, 
            'pos_embed_d': model.pos_embed_d, 
            'hidden_d': model.hidden_d, 
            'win_d': model.win_d, 
            'stoi': model.stoi,
            'itos': model.itos,
        }, 
        'state': model.state_dict()
    }
    torch.save(dict_, ckpt_path)

_ema_loss = None
_ema_alpha = 0.001 
def ema(loss: float) -> float: 
     return _ema_alpha * loss + (1-_ema_alpha) * _ema_loss

if __name__ == '__main__': 
    _handle_args()
    torch.manual_seed(2004)
    meta = _get_meta()
    model = Model(embed_n, embed_d, block_size, pos_embed_d, hidden_d, win_d, meta['stoi'], meta['itos'])
    optimizer = model.configure_optimizer(lr)

    for i in range(steps): 
        optimizer.zero_grad() # backrpop on 1 sample 
        x, y = _get_sample('train')
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        

        _ema_loss = loss.item() if _ema_loss is None else _ema_loss
        _ema_loss = ema(loss.item())
        if i % logging_steps == 0: 
            print(f'loss, step {i}: {loss.item():.4f} ema loss: {_ema_loss:.4f}')

    model.eval()
    losses = torch.zeros(eval_steps)
    with torch.no_grad():
        for i in range(eval_steps):
            x, y = _get_sample('val')
            logits, loss = model(x, y)
            losses[i] = loss.item()
    print(f'eval set loss: {losses.mean():.4f}')

    _save_model(model)
