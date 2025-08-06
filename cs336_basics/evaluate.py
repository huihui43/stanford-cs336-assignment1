# decode and generate given a trained model
import torch
import numpy as np
import random
from module import Transformer_LM
from Tokenizer import Tokenizer
import argparse
from configs import build_config
from pdb import set_trace as T



def decode(vocab_size, model, tokenizer, prompts, max_generate_tokens, temperature_value, p):
    """
    promts: str
    max_generate_tokens: max number of generated tokens
    """
    with torch.no_grad():
        assert p > 0, f"Invalid p {p}"
        x0 = torch.tensor(tokenizer.encode(prompts)).type(torch.uint8) # on cpu
        x0 = x0.unsqueeze(0) # (1, seq_len)
        cnt = 0
        while cnt < max_generate_tokens:
            preds = model(x0) # (1, seq_len, vocab_size)
            logits = preds[0, -1] # (vocab_size,)
            prob_q = torch.exp(logits/temperature_value) # exp(l_i/t)
            sum_logits = torch.sum(prob_q).item()
            prob_q /= sum_logits # normalize

            # top-p sampling
            next_token = top_p_sampling(prob_q, p)
            if next_token == 0: # reach end of text
                break
            else:
                x0 = torch.cat([x0, torch.tensor([[next_token]])], dim=1) 
           
            cnt += 1 
                    
        # decode
        x0 = x0.squeeze(0).tolist()

    return tokenizer.decode(x0)

def top_p_sampling(prob_q, p):
    
    # top-p sampling
    sorted_tensor, indices = torch.sort(prob_q, descending=True)
    indices = indices.tolist()
    sorted_tensor = sorted_tensor.tolist()
    cur_sum = 0
    r = 0
    V_j = [] # record absolute position in vocab
    while cur_sum < p:
        cur_sum += sorted_tensor[r]
        V_j.append(indices[r])
        r += 1

    sum_Vj = 0
    prob_q1 = torch.zeros_like(prob_q)
    for ele in V_j:
        sum_Vj += prob_q[ele].item()
    for ele in V_j:
        prob_q1[ele] = prob_q[ele].item() / sum_Vj 

    next_token = torch.multinomial(prob_q1, num_samples=1).item()

    return next_token



if __name__ == '__main__':


    src = '/home/huihui43/project/stanford-cs336-assignment1/result/train/gpt2_small_exp2/ckpt_final.pkl'
    obj = torch.load(src)

    parser = argparse.ArgumentParser(description="LLM Training")
    parser.add_argument("--exp", help="choose a experiment to do")
    cfg = parser.parse_args()
    args = build_config(cfg.exp)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)



    # load model
    model = Transformer_LM(args.vocab_size,args.context_length,args.d_model,
                           args.num_layers,args.num_heads,args.d_ff,
                           args.rope_theta, device='cpu', dtype=torch.float32)

    model.load_state_dict(obj['model'])
    #model = model.to('cpu')

    # load tokenizer
    tokenizer = Tokenizer.from_files(vocab_filepath=args.vocab_filepath, merges_filepath=args.merges_filepath,special_tokens=['<|endoftext|>'])
    
    prompts = "Once upon a time there was a little boy named Ben."
    max_generate_tokens = 200

    temperature = 0.9
    p = 0.95

    print(decode(args.vocab_size, model, tokenizer, prompts, max_generate_tokens, temperature, p))
