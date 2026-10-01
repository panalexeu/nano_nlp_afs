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
        self.embed_n = embed_n
        self.embed_d = embed_d  
        self.hidden_d = hidden_d 
        self.win_d = win_d 
        
        self.pad_token = embed_n
        self.pad_s = int(win_d/2) 
        self.emb = nn.Embedding(embed_n+1, embed_d) 
        # self.conv = nn.Conv1d()


    def forward(self, ids, targets=None): 
        # s - seq. len.
        # ids shape [s] 
        x = nn.functional.pad(ids, (self.pad_s, self.pad_s), mode='constant', value=self.pad_token)
        x = self.emb(x) # [s, embed_d]
        return x 
