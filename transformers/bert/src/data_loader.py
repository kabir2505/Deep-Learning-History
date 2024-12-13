import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchtext
import scipy
import re
import random
import math
import torch.nn as nn
import numpy
from config import config


text = (
       "Hey, what's up? I'm Jack\n"
       "Hey Jack, I'm Emily. Nice to meet you\n"
       "Nice to meet you too. How's it going?\n"
       'Pretty good! I just got a new job\n'
       'Wow, congrats, Emily!\n'
       'Thanks, Jack!'
   )

sentences=re.sub("['.,!?-]","",text.lower()).split("\n") #remove punctuation & to lower
word_list=list(set(" ".join(sentences).split())) # vocab

word_dict={'[PAD]':0,'[CLS]':1,'[SEP]':2,'[MASK]':3}
for i,w in enumerate(word_list):
    word_dict[w]=i+4

number_dict = {i: w for i, w in enumerate(word_dict)}
vocab_size=len(word_dict)

token_list=list()
for sentence in sentences:
    arr=[word_dict[word] for word in sentence.split(" ")]
    token_list.append(arr)

batch_size=config["batch_size"]
max_pred=config["max_pred"]
max_len=config["max_len"]
def make_batch():
    batch=[]
    pos,neg=0,0

    while pos!=batch_size/2 or neg!=batch_size/2:
        tokens_a_index=random.randrange(len(sentences))
        tokens_b_index=random.randrange(len(sentences))

        tokens_a=token_list[tokens_a_index]
        tokens_b=token_list[tokens_b_index]

        input_ids=[word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict['[SEP]']]
        segment_ids=[0]*(len(tokens_a) + 2) + [1]*(len(tokens_b) + 1)
        
        #Masked language modeling
        n_pred= min(max_pred,max(1,int(round(0.15*len(input_ids)))))
        cand_masked_pos =[i for i,token in enumerate(input_ids) if token!= word_dict['[CLS]'] and token != word_dict['[SEP]']]#positions in input_ids that are not [CLS] or [SEP]
        #randomize the list of cand pos
        random.shuffle(cand_masked_pos)
        masked_tokens,masked_pos=[],[]
        
        for pos in cand_masked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            
            if random.random() < 0.8:
                input_ids[pos]=word_dict['[MASK]']
            elif random.random() < 0.5:  # 10%
                index = random.randint(0, vocab_size - 1)
                input_ids[pos] = word_dict[number_dict[index]]
                
        n_pad = max_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
        

        if tokens_a_index + 1==tokens_b_index and pos  < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True]) # IsNext
            pos+=1
        elif tokens_a_index + 1 != tokens_b_index and neg < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False]) # NotNext
            neg+=1
    return batch