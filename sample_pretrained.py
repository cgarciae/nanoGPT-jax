"""
Sample from a trained model
"""
import os
import pickle
import optax
import tiktoken
from model import GPTConfig, GPT
from pathlib import Path
from flax.training import checkpoints, train_state
from flax import serialization
import jax.numpy as jnp
import jax
import nnx

from utils import print_compiling

# -----------------------------------------------------------------------------
start = "\n"  # or "<|endoftext|>" or whatever you like
num_samples = 10  # number of samples to draw
max_new_tokens = 500  # number of tokens generated in each sample
temperature = 0.8  # higher temperature (up to 1) is more random, lower (down to 0) means more greedy
top_k = (
    200  # retain only the top_k most likely tokens, clamp others to have 0 probability
)
seed = 1337
model_type = "gpt2"  # 'gpt2' or 'gpt2-medium' or 'gpt2-large' or 'gpt2-xl'
exec(open("configurator.py").read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

# model
override_args = dict(dropout=0.0)
model = GPT.from_pretrained(model_type, override_args).split()

# load tokenizer
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
start_ids = encode(start)
x = jnp.array(start_ids, dtype=jnp.int32)[None]
key = jax.random.PRNGKey(seed)


@jax.jit
@print_compiling
def _sample(model: nnx.PureModule[GPT], key, tokens) -> jax.Array:
    return model.merge().generate(
        key,
        tokens,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        temperature=temperature,
    )


def sample(model, key, tokens) -> str:
    tokens = _sample(model, key, tokens)
    return decode(tokens[0])


# run generation
for k in range(num_samples):
    step_key = jax.random.fold_in(key, k)
    sample_str = sample(model, step_key, x)
    print(sample_str)
    print("---------------")
