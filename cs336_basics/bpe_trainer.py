from __future__ import annotations

import time
import os
import regex as re
from pretokenizer import pretokenizer
from typing import BinaryIO
from collections import defaultdict
from logger import setup_logger
from pdb import set_trace as T


"""
计算相邻pair出现的freq
"""
def compute_pair_freqs(word_freq, splits):

    pair_freqs = defaultdict(int)
    for word in word_freq:
        split = splits[word]
        for i in range(len(split) - 1):
            pair_freqs[split[i], split[i+1]] += word_freq[word]
   
    return pair_freqs


# 将best的pair更新到splits里面
def merge_pair(best, index, word_freq, splits, pair_freq):
    """
    pair_freq: {(115,116): 1} 
    """
    a, b = best
    for word in word_freq:
        split = splits[word] 
        if len(split) == 1:
            continue
        
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i+1] == b: # 找到匹配

                # update pair_freq
                pair_freq[a,b] -= word_freq[word]
                if i > 0:
                    pair_freq[split[i-1], a] -= word_freq[word]
                    pair_freq[split[i-1], index] += word_freq[word]
                if i < len(split) - 2:
                    pair_freq[b, split[i+2]] -= word_freq[word]
                    pair_freq[index,split[i+2]] += word_freq[word]

                # update split
                split = split[:i]  + [index] + split[i+2:]

            else:
                i+= 1

        splits[word] = split

    return splits, pair_freq


def find_max(pair_freqs, vocab):

    max_val = max(pair_freqs.values())
    max_list = [key for key, value in pair_freqs.items() if value == max_val]
    max_list2 = [(vocab[key[0]], vocab[key[1]]) for key in max_list]

    return max(list(zip(max_list2, max_list)), key=lambda x:x[0])

def bpe_encoding_per_chunk(
    chunk:str,
    vocab_size:int,
    special_tokens:list[str],
    pretokenizer:object,
    **kwargs
)->list[tuple[bytes, bytes]]:

    n_special = len(special_tokens) 
    merges = []

    VOCAB = defaultdict(int)
    for i in range(256):
        VOCAB[i] = i.to_bytes()

    #=================================================
    # step1: remove all the special tokens 
    # remove special tokens in the chunk
    #pattern = re.escape("|".join(special_tokens))

    chunks = pretokenizer.remove_special(chunk)

    #=================================================
    # step2: calculate word freq and splits
    word_freq = defaultdict(int)
    splits = defaultdict(list)
    for ele in chunks:
        for token in pretokenizer.finditer(ele):
            word = token.group()
            word_freq[word] += 1
            if word not in splits:
                splits[word] = list(word.encode("utf-8")) 

    # compute pair freq counts
    pair_freqs = compute_pair_freqs(word_freq, splits)

    #=================================================
    # step3: do looping
    index = 256 + n_special
    
    while 256 + n_special + len(merges) < vocab_size:

        # 计算相邻pair的频率
        best_byte, best = find_max(pair_freqs,VOCAB) # 比如说是(115, 116)

        # 合并 best, update splits
        splits, pair_freqs = merge_pair(best, index, word_freq, splits, pair_freqs)

        # 更新词汇表
        VOCAB[index] = best_byte[0] + best_byte[1]

        # debug
        merges.append(best_byte)
        index += 1


    return merges


def trainer(
        input_path:str |os.PathLike, 
        vocab_size: int, 
        special_tokens: list[str],
        **kwargs,
    )->tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    main function
    """
    
    # read file as binary 
    num_chunks = 1
    pre_tokenizer = pretokenizer(special_tokens) 

    ff = open(input_path, "rb") 
    # cut into chunks
    boundaries = pre_tokenizer.find_chunk_boundaries(ff, num_chunks)
    merges = []
    vocab = {}


    # serial fetch chunk
    # TODO change it to multi-processing
    ii = 0
    with open(input_path, "rb") as f:    
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            print(f"Deal with chunk {ii+1}")
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            merges_chunk = bpe_encoding_per_chunk(chunk, vocab_size, special_tokens, pretokenizer=pre_tokenizer) 
            ii += 1
            merges += merges_chunk
        
    
    # add special tokens to vocab 
    for ii, ele in enumerate(special_tokens):
        vocab[ii] = special_tokens[ii].encode("utf-8")

    # add 256 Byte chars
    curr_vocab_size = len(vocab)
    for i in range(256):
        vocab[curr_vocab_size + i] = i.to_bytes()
 
    
    # add merges to vocab
    curr_vocab_size = len(vocab)
    n_merges = max(0, vocab_size - curr_vocab_size)
    for ii, ele in enumerate(merges[:n_merges]):
        tmp = list(ele)
        vocab[curr_vocab_size+ii] = tmp[0] + tmp[1]
    
    return vocab, merges[:n_merges]

   



if __name__ == '__main__':

    import cProfile

    fpath = "./tests/fixtures/corpus.en"
    #fpath = "./tests/fixtures/tinystories_sample_5M.txt"
    #fpath = "./data/mySimpleTest.txt"
    #fpath = "./data/mytest.txt"

    logger = setup_logger("test_merge", "./logs/test_merge_multiprocess.log")
    vocab, merges = trainer(fpath,500,["<|endoftext|>"], num_chunks=1)
    logger.info('normal\n')
    logger.info(merges)

    #print(merges)
    '''
    boundaries = find_chunk_boundaries_warp(fpath, 1)
    special_tokens = ["<|endoftext|>"]
    
    # serial fetch chunk
    # TODO change it to multi-processing
    ii = 0
    with open(fpath, "rb") as f:    
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            print(f"Deal with chunk {ii+1}")
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            t1 = time.time()
            merges_chunk = BPE_encoding_per_chunk_re(chunk, 1000, special_tokens) 
    '''


