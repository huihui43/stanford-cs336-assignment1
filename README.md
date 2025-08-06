# CS336 Spring 2025 Assignment 1: Basics

My implementation for assignment 1. 

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

In this assignment, you will go through the following tests:

* uv run pytest tests/test_train_bpe.py
* uv run pytest tests/test_tokenizer.py
* uv run pytest -k test_linear
* uv run pytest -k test_embedding
* uv run pytest -k test_rmsnorm
* uv run pytest -k test_swiglu
* uv run pytest -k test_rope
* uv run pytest -k test_softmax_matches_pytorch
* uv run pytest -k test_scaled_dot_product_attention
* uv run pytest -k test_4d_scaled_dot_product_attention
* uv run pytest -k test_multihead_self_attention
* uv run pytest -k test_transformer_block
* uv run pytest -k test_transformer_lm
* uv run pytest -k test_cross_entropy
* uv run pytest -k test_adamw
* uv run pytest -k test_get_lr_cosine_schedule
* uv run pytest -k test_gradient_clipping
* uv run pytest -k test_get_batch
* uv run pytest -k test_checkpointing














### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

