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
from model import MultiHeadedAttention,PositionwiseFeedForward,PositionalEncoding,DecoderLayer,Decoder,GPT,Embeddings,Generator,subsequent_mask
from tqdm import tqdm
from data_loader import TextDataset,dataloader,vocab
def make_model(vocab,N=12,d_model=768,d_ff=3072,h=12,dropout=0.1):
    """
    Construct a GPT model.

    Args:
        vocab (torchtext.vocab.Vocab): Vocab object from torchtext.

    Keyword Args:
        N (int, optional): Number of Decoder layers. Defaults to 12.
        d_model (int, optional): The dimensionality of the input and output features. Defaults to 768.
        d_ff (int, optional): The dimensionality of the inner layer. Defaults to 3072.
        h (int, optional): The number of heads in the MultiHeadedAttention. Defaults to 12.
        dropout (float, optional): The dropout rate to be applied after the positional encoding and the output of each layer. Defaults to 0.1.

    Returns:
        torch.nn.Module: A GPT model.
    """
    c=copy.deepcopy
    attn=MultiHeadedAttention(h,d_model)
    ff=PositionwiseFeedForward(d_model,d_ff,dropout)
    position=PositionalEncoding(d_model,dropout)
    model=GPT(
        Decoder(DecoderLayer(d_model,c(attn),c(ff),dropout),N),
        nn.Sequential(Embeddings(d_model,vocab),c(position)),
        Generator(d_model,vocab))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class Batch:
    def __init__(self, src, pad=0):
        #src->[batch_len,seq_len]
        
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2) #[batch_size, 1, seq_len]

        # Derive tgt and tgt_y from src
        self.tgt = src[:, :-1]  # Input to the decoder (shift right) [batch_len,seq_len-1]
        self.tgt_y = src[:, 1:]  # Target for loss computation (shift left) [batch_len,seq_len-1]

        # Create tgt_mask
        self.tgt_mask = self.make_std_mask(self.tgt, pad)
        self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """Creates a mask to avoid attention on padding tokens."""
        tgt_mask = (tgt != pad).unsqueeze(-2)  # Shape: [batch_size, 1, seq_len]
        print((tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)).shape)
        return tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data) #[batch_size, seq_len, seq_len]
    
    

class TrainState:
    step:int=0 #steps in the current epoch
    accum_step:int=0 #number of gradient accumulation steps
    samples:int=0 #total #examples used
    tokens:int=0 # total # of tokens processed

pad_token=0
criterion=nn.CrossEntropyLoss(ignore_index=pad_token)

class SimpleLossCompute:
    def __init__(self,generator,criterion,pad):
        """
        Initialize the SimpleLossCompute module.

        Args:
            generator (torch.nn.Module): An nn.Module that maps the output of the decoder to a probability distribution over the vocabulary.
            criterion (torch.nn.Module): The loss function to use. Should be an nn.Module that takes two inputs (logits, labels) and returns a scalar loss.
            pad (int): The padding token id.
        """
        self.generator=generator
        self.criterion=criterion

    def __call__(self,x,y,norm):
        logits=self.generator(x)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
          print("NaN or Inf detected in model outputs")

        loss=self.criterion(
            logits.contiguous().view(-1,logits.size(-1)),
            y.contiguous().view(-1)
        ) / norm
        print('simple',logits.contiguous().view(-1,logits.size(-1)).shape, y.contiguous().view(-1).shape)
        return loss.data*norm,loss

def run_epoch(data_iter,model,loss_compute,optimizer,mode="train",accum_iter=1,train_state=TrainState()):
    """Trains a single epoch"""

    start=time.time()
    total_tokens=0
    total_loss=0
    tokens=0
    n_accum=0

    for i,batch in enumerate(data_iter):
        if torch.isnan(batch.tgt).any() or torch.isinf(batch.tgt).any():
          print("NaN or Inf detected in model inputs")
        assert batch.tgt.shape==batch.tgt_y.shape
        out=model.forward(batch.tgt,batch.tgt_mask)

        loss,loss_node=loss_compute(out,batch.tgt_y,batch.ntokens)
        if mode=="train" or mode=="train + log":
            loss_node.backward()
            for name, param in model.named_parameters():
              if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                  print(f"NaN or Inf in gradient for {name}")
            train_state.step+=1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens


            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_accum+=1
            train_state.accum_step+=1

        total_loss+=loss
        total_tokens+=batch.ntokens
        tokens+=batch.ntokens
        if i%160==0 and (mode=="train" or mode=="train+log"):
            # input_text = tokenizer.decode(batch.src[0].tolist())  # Decode input batch
            # target_text = tokenizer.decode(batch.tgt_y[0].tolist())  # Decode target batch
            # predicted_text = tokenizer.decode(torch.argmax(out, dim=-1)[0].tolist())  # Decode model prediction



            lr=optimizer.param_groups[0]["lr"]
            elapsed=time.time() - start
            print(
                (
                  "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
#                 % (i, n_accum, loss, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0

    return total_loss / total_tokens, train_state



def train_model(
    model,
    train_dataloader,
    optimizer,
    loss_compute,
    epochs,
    accum_iter,
    device,
    train_state=TrainState()
):
    """Trains the model and tracks progress with tqdm."""
    train_history = []

    # Inspect the first batch for debugging shapes
    for i, src in enumerate(train_dataloader):
        batch = Batch(src.to(device), pad=0)
        print(f"src: {batch.src.shape}, tgt: {batch.tgt.shape}, tgt_y: {batch.tgt_y.shape}, tgt_mask: {batch.tgt_mask.shape}")
        break

    # Training loop with tqdm
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} - Training...")
        model.train()

        # Wrap data iterator with tqdm for progress visualization
        batch_iterator = tqdm(
            (Batch(batch.to(device), pad=0) for batch in train_dataloader),
            desc=f"Epoch {epoch + 1}/{epochs}",
            total=len(train_dataloader),
            unit="batch",
        )

        # Train for one epoch
        train_loss, train_state = run_epoch(
            batch_iterator,
            model,
            loss_compute,
            optimizer=optimizer,
            mode="train",
            accum_iter=accum_iter,
            train_state=train_state
        )

        train_history.append(train_loss)
        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}")

    return train_history

def generate_next_words(model, word2idx, input_seq, max_len=30, device='cpu'):
    """
    Generates the next words based on the input sequence using the model.

    Args:
        model: The trained language model.
        tokenizer: Tokenizer used for encoding/decoding sequences.
        input_seq (str): The input sequence to generate text from.
        max_len (int): Maximum number of tokens to generate.
        device (str): Device to run inference on ('cpu' or 'cuda').

    Returns:
        str: The generated text sequence.
    """
    model.eval()

    # Encode the input sequence and move to the device
    #input_ids = torch.LongTensor(tokenizer.encode(input_seq)).unsqueeze(0).to(device)
    #print(torch.LongTensor([word2idx[word] for word in input_seq.split() if word in word2idx]).unsqueeze(0).to(device))
    input_ids = torch.LongTensor([word2idx[word] for word in input_seq.split() if word in word2idx]).unsqueeze(0).to(device)  # Shape: [1, seq_len]
    generated = input_ids  # Start the generated sequence with input

    for _ in range(max_len):
        # Generate predictions for the next token
        with torch.no_grad():
            seq_len = generated.size(1)  # Current sequence length
            tgt_mask = subsequent_mask(seq_len).to(device)  # Shape: [seq_len, seq_len]
            output = model(generated, tgt_mask)  # Model output: [1, seq_len, vocab_size]

        # Take the token with the highest probability
        next_token_logits = output[:, -1, :]  # [batch_size, vocab_size]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)  # Shape: [1, 1]
        temperature = 0.8
        next_token_logits = next_token_logits / temperature
        probabilities = torch.nn.functional.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probabilities, num_samples=1)
        # Append to the sequence
        generated = torch.cat([generated, next_token_id], dim=1)

        # Stop if EOS token is generated
        # if next_token_id.item() == tokenizer.eos_token_id:
        #     break
    indices = generated[0].tolist()  # Assuming we want to decode the first sequence in the batch

    # Map each index to its corresponding word
    words = [idx2word[idx] for idx in indices if idx in idx2word]

    # Join the words into a single string
    generated_text = ' '.join(words)
    return generated_text
