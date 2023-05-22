# %%
"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run in debug mode example:
$ python train.py --batch_size=32 --other=args

To run DDP on 4 gpus on one node, example:
$ torchrun --standalone --nproc_per_node=4 train.py
"""

import os
from pathlib import Path
import time
import math
import pickle
from typing import Tuple
from flax.training import train_state
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import checkpoints
from flax import serialization
import orbax.checkpoint as orbax
import tiktoken
from flax.training import orbax_utils
import nnx

from model import GPTConfig, GPT
from utils import print_compiling

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False  # disabled by default
wandb_project = "owt"
wandb_run_name = "gpt2"  # 'run' + str(time.time())
# data
dataset = "shakespeare"
batch_size = 12
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.95
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = "nccl"  # 'nccl', 'gloo', etc.
# system
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = "bfloat16"  # 'float32' or 'bfloat16'
compile = True  # use PyTorch 2.0 to compile the model to be faster
# sampling
max_new_tokens = 100  # number of tokens generated in each sample
temperature = 0.8  # higher temperature (up to 1) is more random, lower (down to 0) means more greedy
top_k = (
    200  # retain only the top_k most likely tokens, clamp others to have 0 probability
)
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------
checkpoint_path = Path(out_dir, "checkpoint")
checkpoint_manager = orbax.CheckpointManager(
    checkpoint_path,
    checkpointers=orbax.Checkpointer(orbax.PyTreeCheckpointHandler()),
    options=orbax.CheckpointManagerOptions(
        max_to_keep=2,
        # best_fn=lambda c: c['val_loss'],
        # best_mode='min',
        keep_checkpoints_without_metrics=False,
        create=True,
    ),
)
# poor man's data loader, TODO evaluate need for actual DataLoader
data_dir = os.path.join("data", dataset)
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = np.random.randint(len(data) - block_size, size=(batch_size,))
    x = np.stack([data[i : i + block_size].astype(np.int32) for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block_size].astype(np.int32) for i in ix])
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, "meta.pkl")
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta["vocab_size"]
    print(f"vocab_size = {vocab_size} (from {meta_path})")
else:
    print(f"vocab_size not found in {meta_path}, using GPT-2 default of 50257")
    vocab_size = 50257

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    dropout=dropout,
    vocab_size=vocab_size,
)
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    ctx = nnx.Context(jax.random.PRNGKey(0))
    model = GPT(gptconf, ctx=ctx)
    # initialize weights
    state = model.create_state(**config)
    params = state.params
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    latest_step = checkpoint_manager.latest_step()
    assert latest_step is not None, "no checkpoint found"
    checkpoint = checkpoint_manager.restore(latest_step, items=None)
    checkpoint_model_args = checkpoint["model_args"]
    for k, v in model_args.items():
        assert checkpoint_model_args[k] == v, "for now"
        # TODO: think through how passed in params should interact with checkpoint params
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    empty_state = jax.eval_shape(lambda: model.create_state(**config))
    state = serialization.from_state_dict(empty_state, checkpoint["state"])
    iter_num = checkpoint["iter_num"] + 1
    best_val_loss = checkpoint["val_loss"]
elif init_from.startswith("gpt2"):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    state = model.create_state(**config)
    # read off and override the GPT sizing model args from the model config
    model_args["n_layer"] = model.config.n_layer
    model_args["n_head"] = model.config.n_head
    model_args["n_embd"] = model.config.n_embd
else:
    raise RuntimeError(f"init_from={init_from} not supported")

modeldef = model.deref()[1]


# %%
@partial(jax.jit, static_argnames=("train",))
@print_compiling
def forward(state, batch, *, train: bool):
    inputs, labels = batch
    rngs = {}
    if train and dropout > 0.0:
        rngs["dropout"] = jax.random.fold_in(jax.random.PRNGKey(0), state.step)

    ctx = nnx.Context(rngs=rngs, flags=dict(deterministic=False))
    return state.apply_fn([state.params])(inputs, targets=labels, ctx=ctx)[0]


@partial(jax.jit, donate_argnums=(0,))
@print_compiling
def train_step(state: train_state.TrainState, batch):
    def loss_fn(params):
        state_ = state.replace(params=params)
        logits, loss = forward(state_, batch, train=True)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return loss, state


def estimate_loss():
    out = {}
    for split in ["train", "val"]:
        losses = np.zeros(eval_iters)
        for k in range(eval_iters):
            batch = get_batch(split)
            logits, loss = forward(state, batch, train=False)
            losses[k] = float(loss)
        out[split] = losses.mean()
    return out


@jax.jit
@print_compiling
def _sample(params, key, tokens) -> jax.Array:
    model = modeldef.reref(params)
    return model.generate(
        key,
        tokens,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        temperature=temperature,
    )


tokenizer = tiktoken.get_encoding("gpt2")


def sample(params, key, tokens) -> str:
    tokens = _sample(params, key, tokens)
    return tokenizer.decode(tokens[0])


# logging
if wandb_log:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# %%
val_batch = get_batch("val")
# training loop
t0 = time.time()

while True:
    if iter_num % eval_interval == 0:
        print("evaluating...")
        sample_str = sample(
            state.params, jax.random.PRNGKey(0), tokens=val_batch[0][0:1, :5]
        )
        print(f"sample: {sample_str}")
        losses = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "loss/train": losses["train"],
                    "loss/val": losses["val"],
                    "lr": float(lr_schedule(iter_num))
                    if callable(lr_schedule)
                    else lr_schedule,
                }
            )
        if iter_num > 0:
            print(f"saving checkpoint to {out_dir}")
            checkpoint = {
                "state": state,
                "model_args": model_args,
                "iter_num": iter_num,
                "val_loss": losses["val"],
                "config": config,
            }
            checkpoint_manager.save(
                step=iter_num,
                items=checkpoint,
                save_kwargs=dict(
                    save_args=orbax_utils.save_args_from_target(checkpoint)
                ),
            )
    if iter_num == 0 and eval_only:
        break

    loss, state = train_step(state, get_batch("train"))

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = (
            loss.item()
        )  # loss as float. TODO CPU-GPU sync: profile, make sure not slow af
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
