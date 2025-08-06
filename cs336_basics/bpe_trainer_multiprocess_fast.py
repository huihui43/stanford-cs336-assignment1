# accelerate bpe_trainer with multiprocessing
# already pass all the test case
# 目前是最优方案

from __future__ import annotations

import os
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
from linkedlist import LinkedList
import time

from pretokenizer import pretokenizer
from collections import defaultdict, Counter
from cs336_basics.helper import setup_logger
from pdb import set_trace as T

# ===============================
# debug helper function

# ===============================


def compute_pair_freqs(word_freq, word_splits):
    """
        计算相邻pair出现的freq
    """
    pair_freqs = defaultdict(int)
    for word in word_freq:
        word_split = word_splits[word] # linkedlist
        cur = word_split.head 
        while cur and cur.next:
            pair_freqs[cur.data, cur.next.data] += word_freq[word]
            cur = cur.next

    return pair_freqs


#@profile
def merge_pair(
        best:tuple[int,int], 
        index:int, 
        word_freq:dict, 
        word_splits:dict, 
        pair_freq:dict,
        word_bytes:dict):
    """
    # 将best pair更新到原来的结果中
    pair_freq: {(115,116): 1} 
    word_freq:dict[str:int]
    splits: dict[str:linkedlist]
    pair_freq:dict[tuple:int]
    word_bytes:dict[str:set]
    """
    a, b = best
    for word in word_freq:

        word_split = word_splits[word] 
        word_byte = word_bytes[word] # dict, 记录word的每个byte出现的次数

        # cheap check
        if a not in word_byte or b not in word_byte:
            continue

        freq = word_freq[word] # freq of the word

        cur = word_split.head
        prev = cur.prev
        nxt = cur.next

        while cur and nxt: # while i < n_split - 1
            if cur.data == a and nxt.data == b: # 找到匹配

                # update pair_freq
                pair_freq[a,b] -= freq
                if prev:
                    c = prev.data
                    pair_freq[c, a] -= freq # delete old merge
                    pair_freq[c, index] += freq # add new merge
                if nxt.next: 
                    c = nxt.next.data
                    pair_freq[b, c] -= freq
                    pair_freq[index,c] += freq

                # update word split
                cur.data = index 
                cur.next = nxt.next
                nxt = cur.next
                if nxt:
                    nxt.prev = cur

                # update word byte
                word_byte[a] -= 1
                word_byte[b] -= 1
                if a in word_byte and word_byte[a] <= 0:
                    del word_byte[a]
                if b in word_byte and word_byte[b] <= 0:
                    del word_byte[b]
                # add index
                if index not in word_byte:
                    word_byte[index] = 1
                else:
                    word_byte[index] += 1
                word_bytes[word] = word_byte

            else:
                # move one step
                cur = cur.next
                prev = cur.prev
                nxt = cur.next


    return word_splits, pair_freq


def find_max(pair_freqs:dict, vocab:dict):

    max_val = max(pair_freqs.values())
    max_list = [((vocab[key[0]], vocab[key[1]]), key) for key, value in pair_freqs.items() if value == max_val]
    return max(max_list, key=lambda x:x[0])

def pretokenize_chunk(
    input_path:str|os.PathLike,
    start:int,
    end:int,
    pretokenizer:object,
    **kwargs
)->list[tuple[bytes, bytes]]:
    """
    worker function 
    """
   
    #print(f"start child process {os.getpid()}")
    # read file
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    #=================================================
    # step1: remove all the special tokens 
    # remove special tokens in the chunk
    #pattern = re.escape("|".join(special_tokens))

    chunks = pretokenizer.remove_special(chunk)

    #=================================================
    # step2: calculate word_freq and splits
    # word_freq: dict{str: int}
    # splits: dict{str:[]}
    word_freq = defaultdict(int)
    word_splits = defaultdict(LinkedList)
    word_bytes = defaultdict(set)
    for ele in chunks:
        for token in pretokenizer.finditer(ele):
            word = token.group()
            word_freq[word] += 1

            if word not in word_splits:
                encoded_word = list(word.encode('utf-8'))
                word_list = LinkedList(encoded_word)
                word_splits[word] = word_list
                word_bytes[word] = dict(Counter(encoded_word))

    # compute pair freq counts
    pair_freqs = compute_pair_freqs(word_freq, word_splits) # pair_freqs: dict{tuple: int}

    #print(f"finish child process {os.getpid()}")
    return (word_freq, word_splits, pair_freqs, word_bytes)


def summarize(
        list_word_freq:list[dict], 
        list_word_splits:list[dict], 
        list_pair_freqs:list[dict],
        list_word_bytes:list[dict]): 

    """
    list_word_freqs:  
    """
    word_freq = defaultdict(int)
    word_splits = defaultdict(LinkedList)
    pair_freqs = defaultdict(int)
    word_bytes = defaultdict(set)

    # word_freq
    for ele in list_word_freq:
        for k, v in ele.items():
            word_freq[k] += v

    # word_splits
    for ele in list_word_splits:
        for k, v in ele.items():
            if k not in word_splits:
                word_splits[k] = v

    # pair_freqs
    for ele in list_pair_freqs:
        for k, v in ele.items():
            pair_freqs[k] += v

    # word_bytes
    for ele in list_word_bytes:
        for k, v in ele.items():
            if k not in word_bytes:
                word_bytes[k] = v

    return word_freq, word_splits, pair_freqs, word_bytes 

# 计算最终的汇总结果
def cal_final_merges(
        special_tokens:list, 
        vocab_size:int, 
        word_freq:dict, 
        word_splits:dict, 
        pair_freqs:dict,
        word_bytes:dict):

    print("calculate final merges")
    n_special = len(special_tokens) 
    merges = []
    
    progress_bar = tqdm(total=vocab_size - 256 - n_special)

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
        word_splits, pair_freqs = merge_pair(
            best, index, word_freq, word_splits, pair_freqs, word_bytes)
        
        # 更新词汇表
        vocab[index] = best_byte[0] + best_byte[1]

        merges.append(best_byte)
        n_merges += 1
        index += 1
        progress_bar.update(1)

    progress_bar.close()
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

    if 'num_chunks' not in kwargs:
        num_chunks = 1024 
    else:
        num_chunks = kwargs['num_chunks']

    print(f"split file into {num_chunks} chunks")
    
    pre_tokenizer = pretokenizer(special_tokens) 

    ff = open(input_path, "rb") 
    boundaries = pre_tokenizer.find_chunk_boundaries(ff, num_chunks)
    merges = []
    vocab = {}


    t1 = time.time()

    #tracemalloc.start() # start tracing memory chunk
    #snapshot1 = tracemalloc.take_snapshot()
    # =====================================
    # using multiprocessing
    if 1:
        num_process = mp.cpu_count()-1
        print(f"start multiprocess with core {num_process}")
        pool = Pool(processes=num_process)
        res = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            # TODO 在多进程中，如何对局部变量进行修改呢？
            res.append(pool.apply_async(pretokenize_chunk,args=(input_path,start,end,pre_tokenizer)))

        pool.close()
        pool.join()
    
        # get all the results
        output = [ele.get() for ele in res]

    # =====================================
    # debug, do not use multiprocessing
    if 0:
        res = []
        f = open(input_path, "rb")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            res.append(pretokenize_chunk(chunk, pre_tokenizer))
        output = res
        f.close()
    # =====================================

    '''
    # debug
    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    print("Top 20 differences")
    for stat in top_stats:
        print(stat)
    tracemalloc.stop()
    '''

    print(f"finish chunk pretokenize, use {time.time() - t1}!")
    # summarize all the results

    list_word_freqs = [ele[0] for ele in output]
    list_word_splits = [ele[1] for ele in output]
    list_pair_freqs = [ele[2] for ele in output]
    list_word_bytes = [ele[3] for ele in output]

    global_word_freq, global_word_splits, global_pair_freqs, global_word_bytes = summarize(list_word_freqs, list_word_splits, list_pair_freqs, list_word_bytes)

    # calculate final merges
    merges = cal_final_merges(special_tokens, 
                              vocab_size, 
                              global_word_freq, 
                              global_word_splits, 
                              global_pair_freqs, 
                              global_word_bytes)

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

    logger = setup_logger("test_merge", "./logs/test_merge_multiprocess.log")
    vocab, merges = trainer(fpath,500,["<|endoftext|>"], num_chunks=50) 
    #logger.info("multiprocess\n")
    #logger.info(merges)


    #print(merges)
    '''
    boundaries = find_chunk_boundaries_warp(fpath, 1)
    special_tokens = ["<|endoftext|>"]
    
    # serial fetch chunk
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