from Tokenizer import Tokenizer
import os
import pickle


if __name__ == '__main__':

    vocab_filepath = '/home/huihui43/project/stanford-cs336-assignment1/result/tinystories/tinystories_vocab.pkl'
    merges_filepath = '/home/huihui43/project/stanford-cs336-assignment1/result/tinystories/tinystories_merges.pkl'
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath,special_tokens=['<|endoftext|>'])

    # read file
    all_ids = []
    corpus_path = '/home/huihui43/project/stanford-cs336-assignment1/data/mySimpleTest.txt'
    with open(corpus_path) as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)

    n_tokens = len(all_ids)

    # get bytes
    with open(corpus_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        n_bytes = f.tell()

    print(f'compression ratio is : {n_bytes/n_tokens}')



