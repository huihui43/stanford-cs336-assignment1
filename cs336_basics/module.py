import math
import torch
import torch.nn as nn
from einops import rearrange, einsum, reduce, repeat

from pdb import set_trace as T

class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        std = math.sqrt(2 / (in_features + out_features))
        l, u = -3 * math.sqrt(std), 3 * math.sqrt(std)

        data = torch.empty(out_features, in_features, device=device, dtype=dtype)
        nn.init.trunc_normal_(data, 0, std, a=l, b=u)

        self.weight = nn.Parameter(data)

    def forward(self,x:torch.Tensor) ->torch.Tensor:
        """
        x: (batch, sequence, d_in) 
        """
        return einsum(x,self.weight, "... d_in, d_out d_in -> ... d_out")
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        num_embeddings: [int] Size of the vocabulary
        embedding_dim: [int] Dimension of the embedding vectors, i.e., d_model
        """
        super().__init__()
        self.num_embeddins = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        data = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(data, 0,1,-3,3)
        self.weight = nn.Parameter(data)

    def forward(self, token_ids:torch.Tensor) -> torch.Tensor:
        """
        torch_ids: (batch, sequence_length) 
        return (batch, sequence_length, d_model)
        """
        
        b, _ = token_ids.shape
        token_ids = token_ids.reshape(-1).to(torch.int)
        return rearrange(self.weight[token_ids], '(b l) d_model->b l d_model',b=b)


class RMSNorm(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 eps: float = 1e-5, 
                 device=None, 
                 dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        data = torch.ones(d_model, device=device, dtype=dtype)
        self.weight= nn.Parameter(data)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        x: (batch, seq_len, d_model)  
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # compute rms
        rms = torch.sqrt(reduce(x ** 2, 'b l d_model->b l', 'mean') + self.eps)
        rms = rearrange(rms, 'b l -> b l 1')
        res = x / rms * self.weight

        return res.to(in_dtype)


class SwiGLU_FFN(nn.Module):

    def __init__(self,d_in, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_in = d_in
        self.d_ff = d_ff
        self.w1 = Linear(in_features=d_in, out_features=d_ff,device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_in,device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_in, out_features=d_ff,device=device, dtype=dtype)

    def silu_forward(self,x:torch.Tensor)->torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self,x:torch.Tensor) ->torch.Tensor:
        """
        x: (batch, sequence, d_model) 
        """
        return self.w2(self.silu_forward(self.w1(x)) * self.w3(x))
    
    def flops(self, l, res):
        """
        l: seq_len 
        """
        flops = 0
        flops += 3 * 2 * (l * self.d_in * self.d_ff) # only count matrix multiplies (add & multiplication) 
        if 'SwiGLU_FFN' not in res:
            res['SwiGLU_FFN'] = flops

        return flops



class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,
                 theta:float,
                 d_k:int,
                 max_seq_len:int,
                 device:torch.device|None=None):
        """
        theta: [float] theta value for RoPE
        d_k: [int] dimension of query and key vectors
        max_seq_len: [int] max seq lenth that will be inputted
        device: device to store the buffer on 
        """
        super().__init__()
        assert d_k % 2 == 0, "model dimension should be multiple of 2"
        self.theta = theta
        self.d_k = d_k

        num_blocks  = d_k // 2 # num of blocks
        multiplier = 1 / (theta ** (2/d_k))
        thetas = torch.tensor([[m * multiplier ** i for i in range(num_blocks)] for m in range(max_seq_len)])

        # calculate sin and cos
        exp1 = torch.exp(1j * thetas) # e_{ix}
        exp2 = torch.exp(-1j * thetas) # e_{-ix}
        cos_data = 0.5* (exp1 + exp2) 
        sin_data = 1j * (exp2 - exp1) / 2
        
        cos_data = cos_data.real # (max_seq_len, num_blocks)
        sin_data = sin_data.real # (max_seq_len, num_blocks)
        cos_data.to(device)
        sin_data.to(device)

        mask = torch.ones(d_k)
        mask[::2] = -1
        mask.to(device)

        self.register_buffer("cos_data", cos_data, persistent=False)
        self.register_buffer("sin_data", sin_data, persistent=False)
        self.register_buffer("mask",mask, persistent=False)

    def rotate_vector(self, x):
        x1 = rearrange(x, '... (n p)->... p n', p = 2)
        x1 = x1.flip(-2)
        return rearrange(x1, '... p n -> ... (n p)') #[x2, x1, x4, x3, ...]

    def forward(self, x, token_positions)->torch.Tensor:
        """
        x: (..., seq_len, d_k) 
        token_positions: (...,seq_len)
        """
        '''
        if len(token_positions.shape) > 1:
            token_positions = token_positions[0] # use the same token position for all batches
        '''

        cos_data_repeat = repeat(self.cos_data[token_positions],'seq_len block_num->seq_len (block_num 2)')
        sin_data_repeat = repeat(self.sin_data[token_positions],'seq_len block_num->seq_len (block_num 2)')

        # rotate x
        x1 = self.rotate_vector(x) * self.mask 

        return x * cos_data_repeat + x1 * sin_data_repeat
        

       

def softmax(x:torch.tensor, i: int):

    # find max
    maxval = torch.max(x, dim=i, keepdim=True).values
    x1 = torch.exp(x - maxval)
    return x1 / torch.sum(x1, dim=i, keepdim=True)



def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q: (batch_size, ..., seq_len, d_k) 
    k: (batch_size, ..., seq_len, d_k) 
    v: (batch_size, ..., seq_len, d_v) 
    mask: (seq_len, seq_len)
    return: (batch_size, ..., d_v)
    """
    factor = math.sqrt(q.shape[-1])
    qk = einsum(q, k, "... n d_k, ... m d_k -> ... n m")/factor
    if mask is not None:
        qk = qk.masked_fill_(~mask, float('-inf'))
    qk = softmax(qk, i = len(qk.shape)-1)
    return einsum(qk, v, "... n m, ... m d_v -> ... n d_v")



class Multihead_Self_Attention(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0, "input dimension shold be multiple of num heads"
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x, rope_layer=None, token_positions=None):
        """
        x: (b, l, d_model) 
        """
        # generate mask
        seq_len = x.shape[-2]
        mask = torch.triu(torch.ones(seq_len, seq_len)).T
        mask = mask.to(x.device)
        mask = mask.to(torch.bool)

        # project
        q = self.q_proj(x) # (b, l, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # split into head
        d_k = self.d_model // self.num_heads
        q = rearrange(q, "b l (h d_k)-> (b h) l d_k", d_k=d_k) # (b * h, l, d_k)
        k = rearrange(k, "b l (h d_k)-> (b h) l d_k", d_k=d_k) # (b * h, l, d_k)
        v = rearrange(v, "b l (h d_k)-> (b h) l d_k", d_k=d_k) # (b * h, l, d_k)

        # apply rope position encoding
        if rope_layer is not None:
            q = rope_layer(q, token_positions)
            k = rope_layer(k, token_positions)

        attn = scaled_dot_product_attention(q, k, v, mask) # (b*h, l, d_k)
        attn = rearrange(attn, "(b h) l d_k -> b l (h d_k)", d_k=d_k, h=self.num_heads)

        return self.output_proj(attn)

    def flops(self, l, res):
        """
        l: seq_len 
        """
        flops = 0
        val = 3 * 2 * (l * self.d_model * self.d_model) # q_proj, k_proj and v_proj
        if 'Multihead_Self_Attention.proj' not in res:
            res['Multihead_Self_Attention.proj'] = val
        flops += val
        val = 2 * self.num_heads * l * l * self.d_k # qk^T 

        if 'Multihead_Self_Attention.qk' not in res:
            res['Multihead_Self_Attention.qk'] = val
        
        flops += val

        val = 2 * self.num_heads * l * l * self.d_k # (qk^T)V
        if 'Multihead_Self_Attention.qkv' not in res:
            res['Multihead_Self_Attention.qkv'] = val

        flops += val

        if __class__.__name__ not in res: 
            res[__class__.__name__] = flops
        
        return flops




# pre norm version
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, **kwargs):
        super().__init__()

        device = kwargs['device'] if 'device' in kwargs else None
        dtype = kwargs['dtype'] if 'dtype' in kwargs else None

        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU_FFN(d_in=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.attn = Multihead_Self_Attention(d_model=d_model,num_heads=num_heads, device=device,dtype=dtype)
    
    def forward(self, x, token_positions=None, rope_layer=None):
        """
        x: (B, l, d_model)
        """
        x = x + self.attn(self.ln1(x), 
                         rope_layer=rope_layer, 
                         token_positions=token_positions)
        return x + self.ffn(self.ln2(x))

    def flops(self, l, res):
        flops = 0
        flops += self.attn.flops(l, res) # attn
        if 'TransformerBlock.attn' not in res:
            res['TransformerBlock.attn'] = flops
        val = self.ffn.flops(l, res)
        if 'TransformerBlock.ffn' not in res:
            res['TransformerBlock.ffn'] = val
        flops += val
        return flops

class Transformer_LM(nn.Module):

    def __init__(self, 
                 vocab_size, 
                 context_length, 
                 d_model,
                 num_layers, 
                 num_heads,
                 d_ff,
                 rope_theta,
                 **kwargs):

        super().__init__()

        device = kwargs.get('device', None) 
        dtype = kwargs.get('dtype', None) 

        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.vocab_size = vocab_size

        # token embedding        
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.rope_layer = RotaryPositionalEmbedding(rope_theta, d_model//num_heads, context_length) # d_k=8, seq_len=12
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size)

        layers = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (b, l) 
        """
        x = self.token_embeddings(x)
        token_positions = torch.arange(x.shape[1]).to(x.device)

        for layer in self.layers:
            x = layer(x, token_positions = token_positions, rope_layer = self.rope_layer)

        x = self.ln_final(x) # (B, l, d_model)
        x = self.lm_head(x) # (B, l, vocab_size)
        return x
    
    def flops(self):
        
        res = {}
        flops = 0
        for layer in self.layers:
            flops += layer.flops(self.context_length, res)
        val = 2 * self.context_length * self.d_model * self.vocab_size
        res['Transformer_LM.lm_head'] = val
        flops += val
        res[__class__.__name__] = flops
        return res




if __name__ == '__main__':

    '''
    model = Transformer_LM(
        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000
    )

    input = torch.randint(low=0, high=10000, size=(1, 256,))
    y = model(input)
    print(y.shape)
    '''
    import time
    t1 = time.time()
    x = torch.randint(0, 100, size=(10,1000))
    #layer = RotaryPositionalEmbedding(10000, 256, 200)
    #layer = new_RoPE(10000, 256, 200)
    layer = Embedding(100, 256)
    for _ in range(10):
        layer(x)

    print(f"use time {(time.time() - t1)/10}")