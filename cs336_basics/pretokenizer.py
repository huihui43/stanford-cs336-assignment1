import regex as re
from pretokenization_example import find_chunk_boundaries
from pdb import set_trace as T



class pretokenizer:

    def __init__(self, split_tokens =None):
        """
        special_tokens: list[str] 
        """
        self.split_tokens = split_tokens 
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" # GPT2 split
        #self.pattern = r"""\S+""" # split by space

    def remove_special(self,chunk):
        
        if self.split_tokens is not None:
            pattern = "|".join(re.escape(token) for token in self.split_tokens)
            chunks = re.split(pattern, chunk)
            return chunks 
        else:
            return [chunk]

    @staticmethod
    def find_chunk_boundaries(f, num_chunks, split_special_token = "<|endoftext|>"):
        try:
            boundaries = find_chunk_boundaries(f, num_chunks, split_special_token.encode("utf-8"))
        finally:
            f.close()

        return boundaries
 


    def finditer(self, chunk):
        return re.finditer(self.pattern, chunk)
        

if __name__ == '__main__':

    tokenizer = pretokenizer(['<|endoftext|>'])
    
    fpath = "./data/mySimpleTest.txt"
    with open(fpath, "rb") as f:
        chunk = f.read().decode("utf-8", errors="ignore")
    print(chunk)

    chunks = tokenizer.remove_special(chunk)
    print(chunks)