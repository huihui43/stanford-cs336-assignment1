# decode and generate given a trained model
import sys
sys.path.insert(0, 'tests')
from common import gpt2_bytes_to_unicode
import torch
import numpy as np
import random
from module import Transformer_LM, softmax
from Tokenizer import Tokenizer
import argparse
from configs import build_config
from pdb import set_trace as T


def decode(model, tokenizer, prompts, max_generate_tokens, temperature_value, p):
    """
    promts: str
    max_generate_tokens: max number of generated tokens
    """
    with torch.no_grad():
        assert p > 0, f"Invalid p {p}"
        x0 = torch.tensor(tokenizer.encode(prompts)).type(torch.long) # on cpu
        x0 = x0.unsqueeze(0) # (1, seq_len)
        cnt = x0.shape[1]
        while cnt < max_generate_tokens:
            preds = model(x0) # (1, seq_len, vocab_size)
            logits = preds[0, -1] # (vocab_size,)
            prob_q = softmax(logits/temperature_value, i=0) # exp(l_i/t)

            # top-p sampling
            next_token = top_p_sampling(prob_q, p)
            x0 = torch.cat([x0, torch.tensor([[next_token]], dtype=torch.long)], dim=1) 
            if next_token == 0: # reach end of text
                break
            cnt += 1 
                    
        # decode
        x0 = x0.squeeze(0).tolist()

    decoded_string = tokenizer.decode(x0)
    return decoded_string


def top_p_sampling(prob_q, p):
    
    # top-p sampling
    sorted_tensor, indices = torch.sort(prob_q, descending=True)
    cur_sum = 0
    r = 0
    while r < len(indices) and cur_sum < p:
        cur_sum += sorted_tensor[r]
        r += 1
    V_j = indices[:r+1]
    
    sum_Vj = 0
    sum_Vj = torch.sum(prob_q[V_j])
    mask = torch.zeros_like(prob_q)
    mask[V_j] = 1
    prob_q1 = prob_q * mask / sum_Vj
    next_token = torch.multinomial(prob_q1, num_samples=1).item()

    return next_token


if __name__ == '__main__':

    #model_path = '/home/huihui43/project/stanford-cs336-assignment1/result/train/gpt2_small_exp1/ckpt_epoch_0.pkl'
    model_path = '/home/huihui43/project/stanford-cs336-assignment1/result/train/gpt2_small_exp4/ckpt_3660.pkl'
    obj = torch.load(model_path)

    parser = argparse.ArgumentParser(description="LLM Training")
    parser.add_argument("--exp", help="choose a experiment to do")
    cfg = parser.parse_args()
    #args = build_config(cfg.exp)
    args = build_config('gpt2_small')
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load model
    model = Transformer_LM(args.vocab_size,args.context_length,args.d_model,
                           args.num_layers,args.num_heads,args.d_ff,
                           args.rope_theta, device='cpu', dtype=torch.float32)

    model = torch.compile(model) 
    model.load_state_dict(obj['model'])
    model.eval()

    # load tokenizer
    tokenizer = Tokenizer.from_files(vocab_filepath=args.vocab_filepath, merges_filepath=args.merges_filepath,special_tokens=['<|endoftext|>'])
    
    prompts = "Once upon a time, "
    max_generate_tokens = 256

    temperature = 0.7
    p = 0.9

    print(decode(model, tokenizer, prompts, max_generate_tokens, temperature, p))

 
