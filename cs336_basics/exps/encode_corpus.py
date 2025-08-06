# encode text file using trained tokenizer and save to 1d array

import os
import sys
sys.path.insert(0, 'cs336_basics')
from Tokenizer import Tokenizer
import numpy as np
import time
from pdb import set_trace as T


if __name__ == '__main__':

    t1 = time.time()
    dataset_name = 'tinystories'
    # trained tokenizer
    vocab_filepath = '/home/huihui43/project/stanford-cs336-assignment1/result/tokenizer/tinystories/tinystories_vocab.pkl'
    merges_filepath = '/home/huihui43/project/stanford-cs336-assignment1/result/tokenizer/tinystories/tinystories_merges.pkl'
    tokenizer = Tokenizer.from_files(vocab_filepath=vocab_filepath, merges_filepath=merges_filepath,special_tokens=['<|endoftext|>'])

    # corpus to encode
    type = 'train'
    corpus_path = f'/home/huihui43/project/stanford-cs336-assignment1/data/raw/TinyStoriesV2-GPT4-{type}.txt'
    save_root = f'/home/huihui43/project/stanford-cs336-assignment1/data/exp/{dataset_name}_tokens_{type}.npy' 
    
    # read file
    all_ids = []
    with open(corpus_path) as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)

    print(f'using time {time.time() - t1}')

    # save all_ids
    res = np.array(all_ids)
    np.save(save_root, res)
    

    if 0:
        # calculate compression ratio
        n_tokens = len(all_ids)
        # get bytes
        with open(corpus_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            n_bytes = f.tell()

        print(f'compression ratio is : {n_bytes/n_tokens}')



