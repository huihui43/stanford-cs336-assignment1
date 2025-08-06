import torch
import numpy as np
import random
import os
from typing import BinaryIO, IO


def data_loading(x:np.array, 
                 batch_size:int, 
                 context_length:int, 
                 device:str):

    """
    loading data by random order
    return 
        input_sequences:
        next_token target:
    """

    seq_len = x.shape[0]
    assert seq_len > context_length, "context_length is larger than total input length"

    X = np.zeros((batch_size, context_length))
    y = np.zeros((batch_size,context_length))
    i = 0
    while i < batch_size:
        # sample a random starting point
        start = random.randint(0, seq_len-context_length-1)
        X[i,:] = x[start:start+context_length]
        y[i,:] = x[start+1:start+1+context_length]
        i += 1
    
    X = torch.from_numpy(X).type(torch.uint8).to(device)
    y = torch.from_numpy(y).type(torch.uint8).to(device)

    return (X, y)

def data_loading_sequence(
                 x:np.array, 
                 start:int,
                 batch_size:int, 
                 context_length:int, 
                 device:str):

    """
    loading data by sequence order
    return 
        input_sequences:
        next_token target:
    """

    seq_len = x.shape[0]
    assert seq_len > context_length, "context_length is larger than total input length"

    X = np.zeros((batch_size, context_length))
    y = np.zeros((batch_size,context_length))
    i = 0
    while i < batch_size:
        # sample a random starting point
        X[i,:] = x[start:start+context_length]
        y[i,:] = x[start+1:start+1+context_length]
        i += 1
        start += context_length
    
    X = torch.from_numpy(X).type(torch.uint8).to(device)
    y = torch.from_numpy(y).type(torch.uint8).to(device)

    return (X, y)

def save_checkpoint(
        model:torch.nn.Module, 
        optimizer:torch.nn.Module, 
        iteration:int, 
        out: str | os.PathLike | BinaryIO | IO[bytes]):

    obj = {'model':model.state_dict(), 'optim':optimizer.state_dict(), 'it':iteration}
    torch.save(obj, out)

def load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model:torch.nn.Module, 
        optimizer:torch.nn.Module):

    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['optim'])

    return obj['it']