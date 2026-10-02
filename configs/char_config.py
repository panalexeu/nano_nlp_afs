# in paper table 5 POS task config (except pos_embed_d, block_size)
embed_n = 65
embed_d = 50
block_size = 256 
pos_embed_d = 14   
hidden_d = 300
win_d = 5
lr = 1e-3 # 1e-2 from the paper causes training instability => lowered lr
min_lr = 1e-4
epochs = 4
steps = 1_003_854 * epochs
eval_steps = 111_540 