from yacs.config import CfgNode as CN

# config definition
_C = CN()

_C.seed = 42
_C.device='cuda:0'

# input file path
_C.train_text_path = ""

# model param
_C.vocab_size = 50257
_C.context_length = 1024
_C.d_model = 768
_C.num_layers = 12
_C.num_heads = 12
_C.d_ff = 6400
_C.rope_theta=10000

# optimizer
_C.lr=1e-3
_C.betas=(0.9,0.95)
_C.weight_decay=0.01
_C.eps = 1e-7

# training setting
_C.resume = ""
_C.model_save_path = './result/train/gpt2_small_exp1'  # the model output dir
_C.epochs = 50 # the train epochs
