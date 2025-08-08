from collections.abc import Iterable 
import time
import math
import torch
from pdb import set_trace as T

# compute cross_entropy
def cross_entropy_loss(preds, targets):
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    B = preds.shape[0]
    maxval = torch.max(preds, dim=1, keepdim=False).values 
    rows = [i for i in range(B)]
    res = torch.log(torch.sum(torch.exp(preds - maxval.unsqueeze(-1)), dim=1)) + maxval - preds[rows, targets]
    return torch.mean(res)

# compute cross_entropy
def CELoss_and_Perplexity(preds, targets, context_length):
    """Given a tensor of inputs and targets, compute the perplexity 
    
    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "seq_len"]): Tensor of shape (seq_len,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    B = preds.shape[0]
    maxval = torch.max(preds, dim=1, keepdim=False).values 
    rows = [i for i in range(B)]
    neg_log = torch.log(torch.sum(torch.exp(preds - maxval.unsqueeze(-1)), dim=1)) + maxval - preds[rows, targets]
    celoss = torch.mean(neg_log)
    
    neg_log = neg_log.reshape(-1, context_length)
    perplexity = torch.mean(torch.exp(torch.mean(neg_log,dim=1)))# avg per sentence

    return celoss, perplexity




class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}") 
        defaults = {"lr": lr} 
        super().__init__(params, defaults)

    def step(self, closure = None):

        loss = None if closure is None else closure() 
        # self.param_groups is a list
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # state for p
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t+1) * grad # modify in-place
                state["t"] = t + 1

        return loss



class AdamW(torch.optim.Optimizer):
    def __init__(self, 
                 params, 
                 lr, 
                 betas, 
                 weight_decay, 
                 eps=10e-8):
        """
        lr: learning rate
        b1, b2: params to control moment estimates
        mu: weight decay params 
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate : {lr}")
        
        defaults = {"lr":lr, 'betas':betas, 'weight_decay':weight_decay, "eps":eps}
        super().__init__(params, defaults)

    def step(self, closure=None):
        
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            b1,b2 = group['betas']
            mu = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                t = state.get('t', 1)
                if 'm' not in state:
                    m = torch.zeros_like(grad, device=grad.device, dtype=grad.dtype)
                else:
                    m = state['m']
                
                if 'v' not in state:
                    v = torch.zeros_like(grad, device=grad.device, dtype=grad.dtype)
                else:
                    v = state['v']
                
                # update m
                m = b1 * m + (1 - b1) * grad
                # update v
                v = b2 * v + (1 - b2) * grad ** 2
                # update learning rate
                lr_t = lr * math.sqrt(1 - b2 ** t) / (1 - b1 ** t)

                # update param
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data *= (1 - lr * mu)

                # update state
                state['t'] = t + 1
                state['m'] = m
                state['v'] = v
                
        return loss
    
    def set_lr(self, new_lr):
        for group in self.param_groups:
            group['lr'] = new_lr
    


def cosine_annealing_scheduler(t, a_max, a_min, tw, tc):

    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        t (int): Iteration number to get learning rate for.
        a_max(float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        a_min(float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        tw(int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        tc(int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """

    lr = 0
    if t < tw:
        lr = (t / tw) * a_max
    elif t > tc:
        lr = a_min
    else:
        lr = a_min + 0.5 * (a_max - a_min) * (1 + math.cos(math.pi * (t-tw)/(tc-tw)))

    return lr


def gradient_clipping(params: Iterable[torch.nn.Parameter], 
                      max_l2_norm:float):
    
    for p in params:
        if p.grad is None:
            continue

        grad = p.grad.data
        grad_norm = torch.sqrt(torch.sum(grad @ grad.T))
        if grad_norm > max_l2_norm:
            p.grad.data = max_l2_norm/(grad_norm + 1e-6) * grad







if __name__ == '__main__':

    '''
    weights = torch.nn.Parameter(5 * torch.randn((10,10)))
    #opt = SGD([weights], lr=1)
    opt = AdamW([weights], lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)

    for t in range(100):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        print(loss.cpu().item())
        loss.backward() # compute grad
        opt.step()
        T()
    '''


    preds = torch.rand(8,5)
    targets = torch.tensor([1,0,2,2,4,1,4,0])
    t1 = time.time()
    for _ in range(100):
        cross_entropy_loss(preds, targets)

    print(f'use time {(time.time() - t1)/100}') 