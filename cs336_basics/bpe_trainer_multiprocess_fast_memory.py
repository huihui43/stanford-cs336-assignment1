# accelerate bpe_trainer with multiprocessing
# using memory efficient
from __future__ import annotations

from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool
import sys
import time
import gc
import os
from io import StringIO
import pstats
import regex as re
import cProfile
from pretokenizer import pretokenizer
from typing import BinaryIO
from collections import defaultdict, Counter
from cs336_basics.helper import setup_logger
from linkedlist import LinkedList, DoublyNode

def profile(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        pr.disable()
        
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).strip_dirs()
        ps.sort_stats('cumulative')
        ps.print_stats(10)
        
        print(f"Profile for {func.__name__}:\n{s.getvalue()}")
        return result
    return wrapper

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"函数 {func.__name__} 运行时间: {execution_time:.4f} 秒")
        return result
    return wrapper


# 移除全局变量，使用队列收集结果
def collect_results(queue, results):
    """从队列收集结果的守护进程"""
    while True:
        result = queue.get()
        if result is None:  # 结束信号
            break
        results.append(result)



def compute_pair_freqs(word_freq, word_splits):
    """计算相邻pair出现的freq"""
    pair_freqs = defaultdict(int)
    for word in word_freq:
        word_split = word_splits[word]
        cur = word_split.head 
        while cur and cur.next:
            pair_freqs[(cur.data, cur.next.data)] += word_freq[word]
            cur = cur.next
    return pair_freqs

def merge_pair(
        best:tuple[int,int], 
        index:int, 
        word_freq:dict, 
        word_splits:dict, 
        pair_freq:dict,
        word_bytes:dict):
    """将best pair更新到原来的结果中"""
    a, b = best
    for word in word_freq:
        word_split = word_splits[word] 
        word_byte = word_bytes[word]

        if a not in word_byte or b not in word_byte:
            continue

        freq = word_freq[word]
        cur = word_split.head
        prev = cur.prev
        nxt = cur.next

        while cur and nxt:
            if cur.data == a and nxt.data == b:
                # 更新pair_freq
                pair_freq[(a, b)] -= freq
                if prev:
                    c = prev.data
                    pair_freq[(c, a)] -= freq
                    pair_freq[(c, index)] = pair_freq.get((c, index), 0) + freq
                if nxt.next: 
                    c = nxt.next.data
                    pair_freq[(b, c)] -= freq
                    pair_freq[(index, c)] = pair_freq.get((index, c), 0) + freq

                # 更新word split
                cur.data = index 
                cur.next = nxt.next
                nxt = cur.next
                if nxt:
                    nxt.prev = cur

                # 更新word byte
                word_byte[a] -= 1
                word_byte[b] -= 1
                if a in word_byte and word_byte[a] <= 0:
                    del word_byte[a]
                if b in word_byte and word_byte[b] <= 0:
                    del word_byte[b]
                word_byte[index] = word_byte.get(index, 0) + 1
                word_bytes[word] = word_byte

            else:
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
)->tuple[dict, dict, dict, dict]:
    """worker function"""
    try:
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

        chunks = pretokenizer.remove_special(chunk)

        word_freq = defaultdict(int)
        word_splits = dict()  # 使用普通dict代替defaultdict提高性能
        word_bytes = dict()
        
        for ele in chunks:
            for token in pretokenizer.finditer(ele):
                word = token.group()
                word_freq[word] += 1

                if word not in word_splits:
                    encoded_word = list(word.encode('utf-8'))
                    word_list = LinkedList(encoded_word)
                    word_splits[word] = word_list
                    word_bytes[word] = dict(Counter(encoded_word))

        pair_freqs = compute_pair_freqs(word_freq, word_splits)
        return (dict(word_freq), word_splits, dict(pair_freqs), word_bytes)
    except Exception as e:
        print(f"处理块时出错: {e}")
        return (dict(), dict(), dict(), dict())

def summarize(
        list_word_freq:list[dict], 
        list_word_splits:list[dict], 
        list_pair_freqs:list[dict],
        list_word_bytes:list[dict]): 
    """汇总所有进程的结果"""
    word_freq = defaultdict(int)
    word_splits = dict()
    pair_freqs = defaultdict(int)
    word_bytes = dict()

    # 汇总word_freq
    for ele in list_word_freq:
        for k, v in ele.items():
            word_freq[k] += v

    # 汇总word_splits
    for ele in list_word_splits:
        for k, v in ele.items():
            if k not in word_splits:
                word_splits[k] = v

    # 汇总pair_freqs
    for ele in list_pair_freqs:
        for k, v in ele.items():
            pair_freqs[k] += v

    # 汇总word_bytes
    for ele in list_word_bytes:
        for k, v in ele.items():
            if k not in word_bytes:
                word_bytes[k] = v

    return word_freq, word_splits, pair_freqs, word_bytes 

def cal_final_merges(
        special_tokens:list, 
        vocab_size:int, 
        word_freq:dict, 
        word_splits:dict, 
        pair_freqs:dict,
        word_bytes:dict):
    """计算最终的合并结果"""
    print("calculate final merges")
    n_special = len(special_tokens) 
    merges = []
    
    progress_bar = tqdm(total=vocab_size - 256 - n_special)

    vocab = defaultdict(int)
    for i in range(256):
        vocab[i] = i.to_bytes()

    index = 256 + n_special
    n_merges = 0
    while 256 + n_special + n_merges < vocab_size:
        if not pair_freqs:  # 防止没有可合并的对时无限循环
            break
            
        best_byte, best = find_max(pair_freqs, vocab)
        word_splits, pair_freqs = merge_pair(
            best, index, word_freq, word_splits, pair_freqs, word_bytes)
        
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
    """主函数"""
    num_chunks = kwargs.get('num_chunks', 1024)
    print(f"split file into {num_chunks} chunks")
    
    pre_tokenizer = pretokenizer(special_tokens) 

    with open(input_path, "rb") as ff:
        boundaries = pre_tokenizer.find_chunk_boundaries(ff, num_chunks)
    
    t1 = time.time()

    # 使用进程池和队列收集结果
    manager = mp.Manager()
    results = manager.list()  # 跨进程共享的列表
    queue = manager.Queue()
    
    # 启动结果收集进程
    collector_process = mp.Process(target=collect_results, args=(queue, results))
    collector_process.daemon = True
    collector_process.start()

    # 启动工作进程
    num_process = min(mp.cpu_count() - 1, num_chunks)
    print(f"start multiprocess with {num_process} cores")
    
    with Pool(processes=num_process) as pool:
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            pool.apply_async(
                pretokenize_chunk,
                args=(input_path, start, end, pre_tokenizer),
                callback=lambda res: queue.put(res)
            )
        
        # 等待所有任务完成
        pool.close()
        pool.join()
    
    # 发送结束信号并等待收集进程完成
    queue.put(None)
    collector_process.join()

    print(f"finish chunk pretokenize, use {time.time() - t1:.2f} seconds!")

    # 汇总结果
    list_word_freqs = [ele[0] for ele in results]
    list_word_splits = [ele[1] for ele in results]
    list_pair_freqs = [ele[2] for ele in results]
    list_word_bytes = [ele[3] for ele in results]

    global_word_freqs, global_word_splits, global_pair_freqs, global_word_bytes = summarize(
        list_word_freqs, list_word_splits, list_pair_freqs, list_word_bytes)
    
    # 计算最终合并
    merges = cal_final_merges(
        special_tokens, vocab_size, 
        global_word_freqs, global_word_splits, 
        global_pair_freqs, global_word_bytes)

    # 构建词汇表
    vocab = {}
    # 添加特殊令牌
    for ii, ele in enumerate(special_tokens):
        vocab[ii] = ele.encode("utf-8")

    # 添加256个字节字符
    curr_vocab_size = len(vocab)
    for i in range(256):
        vocab[curr_vocab_size + i] = i.to_bytes()
 
    # 添加合并结果到词汇表
    curr_vocab_size = len(vocab)
    n_merges = max(0, vocab_size - curr_vocab_size)
    for ii, ele in enumerate(merges[:n_merges]):
        vocab[curr_vocab_size + ii] = ele[0] + ele[1]

    return vocab, merges[:n_merges]

def main():
    fpath = "./tests/fixtures/corpus.en"
    logger = setup_logger("test_merge", "./logs/test_merge_multiprocess.log")
    vocab, merges = trainer(fpath, 500, ["<|endoftext|>"], num_chunks=50)

if __name__ == '__main__':
    # 在Windows系统上需要保护主模块
    if sys.platform.startswith('win'):
        mp.set_start_method('spawn')
    else:
        mp.set_start_method('fork')
        
    main()