import torch
import os
import time
import random
import numpy as np
from cs336_basics.helper import setup_logger, profile
import argparse
from tqdm import trange
from configs import build_config

from module import Transformer_LM
from optim import AdamW, cosine_annealing_scheduler, gradient_clipping, cross_entropy_loss, CELoss_and_Perplexity
from dataloader import data_loading, data_loading_sequence, load_checkpoint, save_checkpoint
import wandb

from pdb import set_trace as T

def validate(model, valid_tokens, vocab_size, batch_size, context_length, device):
    print("start validating")
    model.eval()
    num_tokens = valid_tokens.shape[0] 
    interval = batch_size * context_length
    valid_loss = []
    valid_perplexity = []
    num_steps = num_tokens // interval
    
    for i in trange(0, num_steps):
        X, y = data_loading_sequence(valid_tokens, i*interval, batch_size, context_length, device) # load one sample
        y = y.view(-1)
        preds = model(X)
        batch_valid_loss, batch_valid_perplexity = CELoss_and_Perplexity(preds.view(-1, vocab_size), y, context_length)
        
        valid_loss.append(batch_valid_loss.item())
        valid_perplexity.append(batch_valid_perplexity.item())

    model.train()
    print("finish validating")
    return np.mean(valid_loss), np.mean(valid_perplexity)

@profile
def train(args):

    # start wandb
    run = wandb.init(
        project='cs336_assignment1',
        config={
            "max_lr":args.max_lr,
            "min_lr":args.min_lr,
            "architecture":args.model_name,
            "dataset":args.dataset_name,
            "epochs":args.epochs,
            "seed":args.seed,
            "resume":args.resume,
        },
    )

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # check model save root
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path, exist_ok=True)
    
    # logger
    logger = setup_logger('train', log_file=os.path.join(args.model_save_path, 'train.log'))

    # build model, optimizer
    model = Transformer_LM(args.vocab_size,args.context_length,args.d_model,
                           args.num_layers,args.num_heads,args.d_ff,
                           args.rope_theta, device=args.device, dtype=torch.float32)

    model = torch.compile(model)
    optim = AdamW(model.parameters(), 1e-3, args.betas, args.weight_decay, eps=args.eps)
    
    # send to device
    model.to(args.device)
    
    if len(args.resume):
        curr_iteration = load_checkpoint(args.resume, model, optim)
        logger.info(f'resume from {args.resume}, curr step is {curr_iteration}')
        model.train()
    else:
        curr_iteration = 0

    lr = cosine_annealing_scheduler(curr_iteration, args.max_lr, args.min_lr, args.warmup_steps, args.cosine_steps)
    optim.set_lr(lr)

    # read input file
    all_train_tokens = np.load(args.train_text_path, mmap_mode='r') # 1d np array
    all_valid_tokens = np.load(args.valid_text_path, mmap_mode='r') # 1d np array
    
    num_batches = int(args.total_tokens_processed / (args.batch_size * args.context_length))

    global_step_cnt = 0 
    for i in range(args.epochs):

        epoch_loss = 0.
        for j in trange(num_batches):
            
            if global_step_cnt < curr_iteration: # skip
                global_step_cnt += 1
                continue 

            t1 = time.time()
            optim.zero_grad() 
            X, y = data_loading(all_train_tokens, args.batch_size, args.context_length, args.device)
            y_pred = model(X) # (B, l, vocab_size)
            loss = cross_entropy_loss(y_pred.view(-1, args.vocab_size), y.view(-1))
            loss.backward() # compute gradient
            gradient_clipping(model.parameters(), args.max_l2_norm) # gradient clipping
            optim.step() # update parameter
            
            epoch_loss += loss.cpu().item()

            # logging
            if global_step_cnt % args.log_per_step == 0:
                run.log({"Avg loss per token": epoch_loss/(j+1), "lr":lr})
                logger.info(f"Epoch: {i+1}/{args.epochs} | Step: {j+1}/{num_batches} | Time: {(time.time()-t1):.2f} | lr: {lr:.4e} | Average Loss (per token): {(epoch_loss / (j+1)):.4f}")
            
            # update lr 
            global_step_cnt += 1
            lr = cosine_annealing_scheduler(global_step_cnt, args.max_lr, args.min_lr, args.warmup_steps, args.cosine_steps)
            optim.set_lr(lr)

            # save model
            if global_step_cnt % args.save_per_step == 0:
                save_checkpoint(model, optim, global_step_cnt, os.path.join(args.model_save_path, f'ckpt_{global_step_cnt}.pkl'))

                # validate
                with torch.no_grad():
                    valid_loss, valid_perplexity = validate(model, all_valid_tokens,args.vocab_size,args.batch_size, args.context_length, args.device)
                    run.log({"valid loss per token": valid_loss, "valid perplexity per sample":valid_perplexity})
                    logger.info(f"Valid loss per token: {valid_loss}, Valid perplexity per sample: {valid_perplexity}")


        # save per epoch
        save_checkpoint(model, optim, global_step_cnt, os.path.join(args.model_save_path, f'ckpt_{global_step_cnt}.pkl'))



    wandb.finish()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="LLM Training")
    parser.add_argument("--exp", help="choose a experiment to do")
    args = parser.parse_args()

    print('doing ', args.exp)

    cfg = build_config(args.exp)

    print(cfg)

    train(cfg)