import os 

from model import Model

if __name__ == '__main__': 
    ckpt_path = './ckpt.pt'
    if not os.path.exists(ckpt_path):
        raise IOError(f'there is no saved {ckpt_path}')
    
    model = Model.from_pretrained(ckpt_path)
    breakpoint()
    
