from typing import Self 

import torch 
import torch.nn as nn 

class Model(nn.Module): 
    def __init__(
        self,
        embed_n: int,  
        embed_d: int, 
        pos_embed_n: int, 
        pos_embed_d: int, 
        hidden_d: int, 
        win_d: int, 
        stoi: dict, 
        itos: dict
    ): 
        super().__init__() 
        self.embed_n = embed_n
        self.embed_d = embed_d
        self.pos_embed_n = pos_embed_n   
        self.pos_embed_d = pos_embed_d 
        self.hidden_d = hidden_d 
        self.win_d = win_d 
        self.stoi = stoi
        self.itos = itos 

        self.pad_token = embed_n
        self.pad_s = int(win_d/2) 
        self.emb = nn.Embedding(embed_n+1, embed_d) 
        self.pos_pad = pos_embed_n 
        self.pos_emb = nn.Embedding(pos_embed_n+1, pos_embed_d)
        # red conv formula: https://docs.pytorch.org/docs/2.14/generated/torch.nn.modules.conv.Conv1d.html#conv1d
        self.f_1 = nn.Conv1d(in_channels=self.embed_d+self.pos_embed_d, out_channels=self.hidden_d, kernel_size=self.win_d)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.f_2  = nn.Linear(in_features=hidden_d, out_features=hidden_d*2)
        self.f_3 = nn.Linear(in_features=hidden_d*2, out_features=embed_n)

    def forward(self, ids, targets=None): 
        b,s = ids.shape
        pos = torch.arange(s-1,-1,-1).expand(b,s) # claude suggested distance from the end: last token = 0, the one before = 1
        tok = nn.functional.pad(ids, (self.pad_s, self.pad_s), mode='constant', value=self.pad_token)
        pos = nn.functional.pad(pos, (self.pad_s, self.pad_s), mode='constant', value=self.pos_pad)
        x = torch.cat([self.emb(tok), self.pos_emb(pos)], dim=-1) # [b, padded_s, embed_d + pos_embed_d]
        x = x.transpose(1,2) # [b, embed_d + pos_embed_d, padded_s] cause conv1d expects (batch, channels, len)
        x = self.f_1(x) # [b, hidden_d, s]
        x = self.pool(x) # [b, hidden_d, 1]        
        x = x.squeeze(-1) # [b, hidden_d]
        x = self.f_2(x) # [b, hidden_d*2]
        x = nn.functional.hardtanh(x) 
        logits = self.f_3(x) # [b, vocab]

        loss = None 
        if targets is not None: 
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def configure_optimizer(self, lr): 
        return torch.optim.SGD(self.parameters(), lr=lr)

    @classmethod
    def from_pretrained(cls, ckpt_path='./ckpt.pt') -> Self:
        ckpt = torch.load(ckpt_path)
        model = cls(**ckpt['cfg'])
        model.load_state_dict(ckpt['state'])
        return model

    def encode(self, s: str) -> list[int]: 
        return [self.stoi[ch] for ch in s]

    def decode(self, ids: list[int]) -> str: 
        return ''.join(self.itos[id_] for id_ in ids)
    