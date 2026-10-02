import os 

import torch

from model import Model

if __name__ == '__main__': 
    ckpt_path = './ckpt.pt'
    if not os.path.exists(ckpt_path):
        raise IOError(f'there is no saved {ckpt_path}')
    model = Model.from_pretrained(ckpt_path)
    model.requires_grad_(False)
    model.eval()
    
    # sampling params 
    t = 1.0 
    max_tokens = 2048
    greedy = False

    # decoding
    while True: 
        print('\n' + '*' * 64)
        prompt = input('msg: ')
        token_count = 0
        while token_count < max_tokens: 
            ids = model.encode(prompt)[-model.pos_embed_n:] # seq truncation
            ids_torch = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            logits, _ = model(ids_torch) 
            logits /= t
            probs = torch.softmax(logits, dim=-1)
            token = 2   # ! in char vocab
            if greedy:  # for now only greedy
                token = torch.argmax(probs).tolist()
            else: 
                token = torch.multinomial(probs, num_samples=1).squeeze().tolist()
            ids.append(token)
            prompt = model.decode(ids)
            token_count += 1
            
            print(prompt[-1], end='')