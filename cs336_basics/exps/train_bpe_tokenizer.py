# train tokenizer using bpe algo

import time
import os
import pickle
from tests.adapters import run_train_bpe

#path = './tests/fixtures/tinystories_sample_5M.txt'
path = '/home/huihui43/project/stanford-cs336-assignment1/data/raw/TinyStoriesV2-GPT4-train.txt'
#path = '/home/huihui43/project/stanford-cs336-assignment1/data/raw/owt_train.txt/owt_train.txt'

dataname = 'tinystories'
#dataname = 'openwebtext'

vocab_size = 10000 # for tinystories
#vocab_size = 32000 # for openwebtext

special_tokens = ["<|endoftext|>"]
output_path = './result/tokenizer/'+dataname

if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

t1 = time.time()
vocab, merges = run_train_bpe(path, vocab_size, special_tokens)
print(f"use time {time.time() - t1}")

# save results
with open(os.path.join(output_path, f"{dataname}_vocab.pkl"), "wb") as f:
    pickle.dump(vocab, f)

with open(os.path.join(output_path, f"{dataname}_merges.pkl"), "wb") as f:
    pickle.dump(merges, f)