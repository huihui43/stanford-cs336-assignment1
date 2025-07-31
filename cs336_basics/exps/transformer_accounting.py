import sys
sys.path.insert(0, 'cs336_basics')
from module import Transformer_LM
import pandas as pd

from pdb import set_trace as T

def count_trainable_parameters(model):
    res = []
    for p in model.parameters():
        if p.requires_grad:
            res.append(p.numel())

    return sum(res)


if __name__ == '__main__':

    model = Transformer_LM(
        vocab_size=50257,
        context_length=1024,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=6400,
        rope_theta=10000
    )

    #print(count_trainable_parameters(model)) 
    res_dict = model.flops()
    df = pd.DataFrame(res_dict, index=['S'])
    df.to_csv('./transformer_analysis.csv', mode='a+')

