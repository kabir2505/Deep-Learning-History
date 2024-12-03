import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torch.functional as F
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy,pandas
# Set to False to skip notebook execution (e.g. for debugging)
from data_loader import get_batch,itos,stoi,vocab_size
from model import make_model
from engine import Batch,model,loss_compute, optimizer
from config import device
from tqdm import tqdm

max_iters = 9000
losses = []
save_interval = 1500

import os
if os.path.isfile("model_weights_9000.pth"):
    if device=="cpu":

        model.load_state_dict(torch.load("model_weights_9000.pth", weights_only=True,map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load("model_weights_9000.pth", weights_only=True))
    

else:

    for iter in tqdm(range(max_iters), desc="Training Progress"):

        xb = get_batch("train")
        batch = Batch(xb)
        src = batch.tgt.to(device)
        tgt = batch.tgt_y.to(device)
        print('src',src.shape)
        out = model.forward(src)  # Shape: [batch_size, seq_len, d_model]

        logits = model.generator(out)  # Shape: [batch_size, seq_len, vocab_size]


        loss, loss_node = loss_compute(out, tgt, batch.ntokens.to(device))


        losses.append(loss.item())


        predicted = torch.argmax(logits, dim=-1)  # Predicted indices: [batch_size, seq_len]
        correct_predictions = (predicted == tgt).float().sum().item()
        total_characters = tgt.numel()  # Total characters in the target
        accuracy = (correct_predictions / total_characters) * 100  # Accuracy as a percentage

        tqdm.write(
            f"Iteration {iter + 1}/{max_iters}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%"
        )


        loss_node.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


        if (iter + 1) % save_interval == 0:
            torch.save(model.state_dict(), f"model_weights_{iter+1}.pth")
            print(f"Saved model weights at iteration {iter + 1}")
            


def generate_next_words(model, input_seq, max_len=30, device='cpu'):

    model.eval()

    # Encode the input sequence and move to the device
    #input_ids = torch.LongTensor(tokenizer.encode(input_seq)).unsqueeze(0).to(device)
    #print(torch.LongTensor([word2idx[word] for word in input_seq.split() if word in word2idx]).unsqueeze(0).to(device))
    input_ids = torch.LongTensor([stoi[word] for word in input_seq if word in stoi]).unsqueeze(0).to(device)  # Shape: [1, seq_len]
    generated = input_ids  # Start the generated sequence with input

    for _ in range(max_len):
        # Generate predictions for the next token
        with torch.no_grad():
            seq_len = generated.size(1)  # Current sequence length

            output = model(generated)
            output=model.generator(output)

        next_token_logits = output[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        temperature = 0.8
        next_token_logits = next_token_logits / temperature
        probabilities = torch.nn.functional.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probabilities, num_samples=1)
        generated = torch.cat([generated, next_token_id], dim=1)

        if (generated >= vocab_size).any():
            #print("Generated sequence contains invalid token indices!")
            generated = generated.clamp(0, vocab_size - 1)  # Clamp to valid range
    indices = generated[0].tolist()  # Assuming we want to decode the first sequence in the batch

    # Map each index to its corresponding word
    words = [itos[idx] for idx in indices if idx in itos]

    # Join the words into a single string
    generated_text = ''.join(words)
    return generated_text

input="Silicon Valley"
generate_next_words(model,input,max_len=250,device=device)