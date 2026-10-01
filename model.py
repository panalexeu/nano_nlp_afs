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
        # red conv formula: https://docs.pytorch.org/docs/2.14/generated/torch.nn.modules.conv.Conv1d.html#conv1d
        self.f_1 = nn.Conv1d(in_channels=self.embed_d, out_channels=self.hidden_d, kernel_size=self.win_d)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.f_2  = nn.Linear(in_features=hidden_d, out_features=hidden_d*2)
        self.f_3 = nn.Linear(in_features=hidden_d*2, out_features=embed_n)

    def forward(self, ids, targets=None): 
        # b - batch, size 
        # s - seq. len.
        # ids shape [b, s] 
        x = nn.functional.pad(ids, (self.pad_s, self.pad_s), mode='constant', value=self.pad_token)
        x = self.emb(x) # [b, s, embed_d]
        x = x.transpose(1,2) # [b, embed_d, s] cause conv1d expects (batch, channels, len)
        x = self.f_1(x) # [b, hidden_d, s]
        x = self.pool(x) # [b, hidden_s, 1]        
        x = x.squeeze(-1) # [b, hidden_s]
        x = self.f_2(x) # [b, hidden_s*2]
        x = nn.functional.hardtanh(x) 
        x = self.f_3(x) # [b, embed_n]

        return x 
