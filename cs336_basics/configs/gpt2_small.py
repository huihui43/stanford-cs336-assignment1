from yacs.config import CfgNode as CN
import torch

# config definition
_C = CN()

_C.seed = 42
_C.device='cuda:0'

# input file path
_C.train_text_path = "/home/huihui43/project/stanford-cs336-assignment1/data/exp/tinystories_tokens_train.npy"
_C.valid_text_path = "/home/huihui43/project/stanford-cs336-assignment1/data/exp/tinystories_tokens_valid.npy"

# info
_C.model_name = 'gpt2_small'

# tokenizer path
_C.dataset_name = 'tinystories'
_C.vocab_filepath = '/home/huihui43/project/stanford-cs336-assignment1/result/tokenizer/tinystories/tinystories_vocab.pkl'
_C.merges_filepath = '/home/huihui43/project/stanford-cs336-assignment1/result/tokenizer/tinystories/tinystories_merges.pkl'

# model param
_C.vocab_size = 10000
_C.context_length = 256
_C.d_model = 512
_C.num_layers = 4
_C.num_heads = 16
_C.d_ff = 1344
_C.rope_theta=10000

# optimizer
_C.betas=(0.9,0.95)
_C.weight_decay=0.01
_C.eps = 1e-7
_C.max_l2_norm = 1.0

# scheduler
_C.max_lr = 5e-5
_C.min_lr = 5e-6
_C.warmup_steps = 200
_C.cosine_steps = 1800

# training setting
_C.resume = ""
_C.model_save_path = './result/train/gpt2_small_exp4'  # the model output dir
_C.epochs = 3 # the train epochs
_C.batch_size = 128
#_C.total_tokens_processed = 327680000
_C.total_tokens_processed = 40000000
_C.save_per_step = 500
_C.log_per_step = 100
