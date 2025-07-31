# accelerate bpe_trainer with multiprocessing
from __future__ import annotations

import time
import multiprocessing as mp
from multiprocessing import Pool

import os
from io import StringIO
import pstats
import regex as re
import cProfile
from pretokenizer import pretokenizer
from typing import BinaryIO
from collections import defaultdict
from logger import setup_logger
from pdb import set_trace as T


def profile(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).strip_dirs()
        ps.sort_stats('cumulative')
        ps.print_stats()
        
        print(f"Profile for {func.__name__}:\n{s.getvalue()}")
        return result
    return wrapper


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
        n_split = len(split)

        if n_split == 1:
            continue
        i = 0
        freq = word_freq[word] # freq of the word
        n_split = len(split)
        while i < n_split - 1:

            if split[i] == a and split[i+1] == b: # 找到匹配
                # update pair_freq
                pair_freq[a,b] -= freq
                if i > 0:
                    pair_freq[split[i-1], a] -= freq
                    pair_freq[split[i-1], index] += freq
                if i < n_split - 2:
                    pair_freq[b, split[i+2]] -= freq
                    pair_freq[index,split[i+2]] += freq

                # update split
                split = split[:i]  + [index] + split[i+2:]
                n_split = len(split)

            else:
                i+= 1

        splits[word] = split

    return splits, pair_freq


def find_max(pair_freqs:dict, vocab:dict):

    max_val = max(pair_freqs.values())
    max_list = [((vocab[key[0]], vocab[key[1]]), key) for key, value in pair_freqs.items() if value == max_val]
    return max(max_list, key=lambda x:x[0])


# TODO 改为multi processing的worker函数
def parallel_calculate_per_chunk(
    chunk:str,
    pretokenizer:object,
    **kwargs
)->list[tuple[bytes, bytes]]:

    # read file

    #=================================================
    # step1: remove all the special tokens 
    # remove special tokens in the chunk
    #pattern = re.escape("|".join(special_tokens))

    print(f"process in 子进程 {os.getpid()}, string len: {len(chunk)}, ")

    chunks = pretokenizer.remove_special(chunk)

    #=================================================
    # step2: calculate word_freq and splits
    # word_freq: dict{str: int}
    # splits: dict{str:[]}
    word_freq = defaultdict(int)
    splits = defaultdict(list)
    for ele in chunks:
        for token in pretokenizer.finditer(ele):
            word = token.group()
            word_freq[word] += 1
            if word not in splits:
                splits[word] = list(word.encode("utf-8")) 

    # compute pair freq counts
    pair_freqs = compute_pair_freqs(word_freq, splits) # pair_freqs: dict{tuple: int}

    # TODO 在step2，分chunks分别计算word_freq, splits, 和 pair_freqs， 再合并到一起
    return (word_freq, splits, pair_freqs)

# TODO 将所有进程计算的结果进行汇总
def summarize(
        list_word_freq:list[dict], 
        list_splits:list[dict], 
        list_pair_freqs:list[dict]):

    """
    list_word_freqs:  
    """
    word_freq = defaultdict(int)
    splits = defaultdict(list)
    pair_freqs = defaultdict(int)

    # word_freq
    for ele in list_word_freq:
        for k, v in ele.items():
            word_freq[k] += v

    # splits
    for ele in list_splits:
        for k, v in ele.items():
            if k not in splits:
                splits[k] = v

    # pair_freqs
    for ele in list_pair_freqs:
        for k, v in ele.items():
            pair_freqs[k] += v

    return word_freq, splits, pair_freqs


# TODO: 汇总所有chunk的信息之后，计算最终的merges
#@profile
def cal_final_merges(
        special_tokens:list, 
        vocab_size:int, 
        word_freq:dict, 
        splits:dict, 
        pair_freqs:dict):

    print("calculate final merges")
    n_special = len(special_tokens) 
    merges = []
    
    vocab = defaultdict(int)
    for i in range(256):
        vocab[i] = i.to_bytes()

    #=================================================
    # step3: do looping
    index = 256 + n_special
    n_merges = 0
    while 256 + n_special + n_merges < vocab_size:

        # 计算相邻pair的频率
        best_byte, best = find_max(pair_freqs,vocab) # 比如说是(115, 116)
        # 合并 best, update splits
        splits, pair_freqs = merge_pair(best, index, word_freq, splits, pair_freqs)

        # 更新词汇表
        vocab[index] = best_byte[0] + best_byte[1]

        merges.append(best_byte)
        n_merges += 1
        index += 1


    return merges

@profile
def trainer(
        input_path:str |os.PathLike, 
        vocab_size: int, 
        special_tokens: list[str],
        **kwargs,
    )->tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    main function
    """
    
    if 'num_chunks' not in kwargs:
        num_chunks = mp.cpu_count()-1
    else:
        num_chunks = kwargs['num_chunks']

    pre_tokenizer = pretokenizer(special_tokens) 

    ff = open(input_path, "rb") 
    # cut into chunks
    boundaries = pre_tokenizer.find_chunk_boundaries(ff, num_chunks)
    merges = []
    vocab = {}

    print(f"using multiprocess with core {len(boundaries)}")
    chunks = []
    with open(input_path, "rb") as f:    
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

    with Pool(processes=mp.cpu_count() - 1) as p:
        res = [p.apply_async(parallel_calculate_per_chunk, (chunk, pre_tokenizer)) for chunk in chunks]
       
        p.close()
        p.join() 
        # get all the results
        output = [ele.get() for ele in res]

    print("finish multiprocessing!")
    list_word_freq = [ele[0] for ele in output]
    list_splits = [ele[1] for ele in output]
    list_pair_freqs = [ele[2] for ele in output]

    # summarize all the results
    word_freq, splits, pair_freqs = summarize(list_word_freq, list_splits, list_pair_freqs)

    # calculate final merges
    merges = cal_final_merges(special_tokens, vocab_size, word_freq, splits, pair_freqs)

    # ===========================
    # output
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

   
def main():

    fpath = "./tests/fixtures/corpus.en"
    #fpath = "./tests/fixtures/tinystories_sample_5M.txt"
    #fpath = "./data/mytest.txt"
    #fpath = "./data/mytest.txt"

    vocab, merges = trainer(fpath,500,["<|endoftext|>"], num_chunks=1)

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






if __name__ == '__main__':

    main()