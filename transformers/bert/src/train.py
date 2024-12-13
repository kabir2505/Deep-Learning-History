import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchtext
import scipy
import re
import random
import math
import torch.nn as nn
import numpy
from config import config
from model import BERT
from data_loader import make_batch,vocab_size,max_len,number_dict,text

d_model=config["d_model"]
d_ff=config["d_ff"]
n_heads=config["n_heads"]
d_k=config["d_k"]
d_v=config["d_v"]
n_segments=config["n_segments"]
n_layers=config["n_layers"]

model = BERT(vocab_size=vocab_size,d_model=d_model,max_len=max_len,n_segments=n_segments,h=n_heads,d_ff=d_ff,n_layers=n_layers)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch = make_batch()
batch = make_batch()
input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(*batch))

for epoch in range(40):
    logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
    loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens) # for masked LM
    loss_lm = (loss_lm.float()).mean()
    loss_clsf = criterion(logits_clsf, isNext) # for sentence classification
    loss = loss_lm + loss_clsf
    optimizer.zero_grad()
    if (epoch + 1) % 10 == 0:
        print('Epoch:',(epoch + 1), 'cost =', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()
    

input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(batch[0]))
print(text)
print([number_dict[w.item()] for w in input_ids[0] if number_dict[w.item()] != '[PAD]'])
logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
print('masked tokens list : ',[pos.item() for pos in masked_tokens[0] if pos.item() != 0])
print('predict masked tokens list : ',[pos for pos in logits_lm if pos != 0])

logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
print('isNext : ', True if isNext else False)
print('predict isNext : ',True if logits_clsf else False)