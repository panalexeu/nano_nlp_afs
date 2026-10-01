import torch.nn as nn 

class Model(nn.Module): 
    def __init__(
        self,
        embed_n: int, 
        embed_d: int, 
        hidden_d: int, 
        win_d: int  
    ): 
        super().__init__() 
        self.emb = nn.Embedding(embed_n, embed_d) 

    def forward(self, idx, targets=None): 
        # b - batch size  
        # s - seq. len.
        # idx shape [b, s] 
        x = self.emb(idx)
        return x 
