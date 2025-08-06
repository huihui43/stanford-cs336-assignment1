import os

# debug
import json
import sys
sys.path.insert(0, "./tests/")
from common import FIXTURES_PATH
import tiktoken

from collections.abc import Iterable
from collections import defaultdict 
import pickle
import heapq as hq
import regex as re
from pretokenizer import pretokenizer
from pdb import set_trace as T


def find_all_substrings_regex(string, substring):
    return [match.start() for match in re.finditer(f'(?={re.escape(substring)})', string)]

def split_and_capture(string, specials):
    # 按长度降序排列分隔符，确保较长的模式优先匹配
    specials.sort(key=len, reverse=True)
    # 转义每个分隔符并构建正则表达式
    escaped_specials = [re.escape(s) for s in specials]
    pattern = '(' + '|'.join(escaped_specials) + ')'
    # 执行分割并保留分隔符
    return re.split(pattern, string)

def sum_bytes(bytes:list, i:int, j:int):
    res = b""
    for ele in bytes[i:j+1]:
        res += ele
    return res

class Tokenizer:

    def __init__(self, vocab, merges, special_tokens=None):
        """
        vocab: dict[int, bytes] 
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        """
        self.vocab =vocab # token_to_tocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab_to_tokens = self.create_vocab_mapping()
        self.pre_tokenizer = pretokenizer(special_tokens)

    def create_vocab_mapping(self):
        # create mapping from vocab to token id
        vocab_to_tokens = defaultdict(lambda: -1)
        for key, value in self.vocab.items():
            vocab_to_tokens[value] = key

        return vocab_to_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):

        try:
            with open(vocab_filepath, 'rb') as f:
                vocab =  pickle.load(f)
        except FileNotFoundError:
            print(f"Error: can not find {vocab_filepath}")

        try:
            with open(merges_filepath, 'rb') as f:
                merges=  pickle.load(f)
        except FileNotFoundError:
            print(f"Error: can not find {merges_filepath}")

        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, chunk:str)->list[int]:
        """
        encode per chunk 
        """

        # split on specials and keep specials
        if self.special_tokens is not None:
            chunks = split_and_capture(chunk, self.special_tokens)
        else:
            chunks = [chunk]

        res = []
        merges_dict = {i:v for v, i in enumerate(self.merges)}

        for chunk in chunks:
            if self.special_tokens is not None and chunk in self.special_tokens:
                res.append(self.vocab_to_tokens[chunk.encode('utf-8')])
            else:
                # do pre_tokenize
                for token in self.pre_tokenizer.finditer(chunk): 
                    msg_bytes= token.group().encode('utf-8')   
                    # directly find the token
                    if msg_bytes in self.vocab_to_tokens:
                        res.append(self.vocab_to_tokens[msg_bytes])
                    else:
                        bytes = [m.to_bytes() for m in list(msg_bytes)] 
                        i = 0 # iterate through bytes
                        # merging 
                        pair_cnt = [] # minheap: (merge order, tuples[bytes, bytes])
                        # initialize
                        while i < len(bytes) - 1:
                            pair = (bytes[i], bytes[i+1])
                            if pair in merges_dict:
                                hq.heappush(pair_cnt, (merges_dict[pair], pair))
                            i += 1

                        overlap_pairs = defaultdict(int)
                        # looping
                        while len(pair_cnt):
                            _, pair = hq.heappop(pair_cnt)
                            
                            while pair in overlap_pairs and len(pair_cnt):
                                overlap_pairs[pair] -= 1
                                if overlap_pairs[pair] == 0:
                                    del overlap_pairs[pair]
                                    
                                _, pair = hq.heappop(pair_cnt)

                            if pair in overlap_pairs: # no further merging pair
                                break

                            sum_pair = pair[0] + pair[1]
                            idx = 0
                            while idx < len(bytes) - 1:
                                if bytes[idx] == pair[0] and bytes[idx+1] == pair[1]:
                                    break
                                idx +=1
                            
                            # add overlap pairs
                            if idx > 0:
                                overlap_pairs[bytes[idx-1], bytes[idx]] += 1
                            if idx + 1 < len(bytes) - 1:
                                overlap_pairs[bytes[idx+1], bytes[idx+2]] += 1

                            # update bytes
                            bytes = bytes[:idx] + [sum_pair] + bytes[idx+2:]

                            # update pair_cnt
                            if idx > 0:
                                new_pair = (bytes[idx-1], sum_pair)
                                if new_pair in merges_dict:
                                    hq.heappush(pair_cnt, (merges_dict[new_pair], new_pair))
                            if idx < len(bytes) - 1 and idx != -1:
                                new_pair = (sum_pair, bytes[idx+1])
                                if new_pair in merges_dict:
                                    hq.heappush(pair_cnt, (merges_dict[new_pair], new_pair))
                        
                        # finish merging, encode final bytes
                        for ele in bytes:
                            try:
                                res.append(self.vocab_to_tokens[ele])
                            except ValueError:
                                print(f'tokenizer cannot find token mapping for {ele}')
                                res.append(-1)

        return res


    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        
        batch = ""
        for chunk in iterable:
            batch += chunk
            if sys.getsizeof(batch) >= 1024 * 1024: # more than 1M
                res = self.encode(batch)
                batch = "" # reset batch
                for ele in res:
                    yield ele 
                

        if batch: # 处理最后一批
            res = self.encode(batch)
            for ele in res:
                yield ele


    def decode(self, ids: list[int]) ->str:
        res = b''
        replace = b'\xf0'
        for id in ids:
            if id in self.vocab:
                res += self.vocab[id] 
            else:
                res += replace

        return res.decode('utf-8', errors='replace')

if __name__ == '__main__':


    VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
    MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"

    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, 
        merges_path=MERGES_PATH, 
        special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"]
    )

    #test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    #test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"

    #corpus_path = FIXTURES_PATH / "address.txt"

    '''
    corpus_path = '/home/huihui43/project/stanford-cs336-assignment1/data/mySimpleTest.txt'
    with open(corpus_path) as f:
        corpus_contents = f.read()

    ids = tokenizer.encode(corpus_contents)
    reference_ids = reference_tokenizer.encode(corpus_contents) 

    print(f" ids: {ids}")
    print(f"rids: {reference_ids}")

    decoded_string = tokenizer.decode(ids)
    print(f"decoded_ids: {decoded_string}")

    rdecoded_string = reference_tokenizer.decode(reference_ids)
    print(f"rdecoded_ids: {rdecoded_string}")


    # write to file
    with open('./tttttest.txt', 'wt') as f:

        f.write(" ".join(str(ele) for ele in ids))
        f.write("\n")
        f.write(" ".join(str(ele) for ele in reference_ids))
        f.write("\n")
        f.write(" ".join(tokenizer.decode([ele]) for ele in ids))
        f.write("\n")
        f.write(" ".join(reference_tokenizer.decode([ele]) for ele in reference_ids))
    '''

    # test iterable
    all_ids = []
    with open(FIXTURES_PATH / "tinystories_sample.txt") as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)
    print(all_ids)

