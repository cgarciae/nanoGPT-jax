"""
Sample from a trained model
"""
# %%
from functools import partial
from typing import Tuple
import numpy as np
import tiktoken
from model import GPT, CausalSelfAttention
import jax.numpy as jnp
import jax
import nnx

from utils import print_compiling

# -----------------------------------------------------------------------------
start = "The cat"  # or "<|endoftext|>" or whatever you like
num_samples = 10  # number of samples to draw
max_new_tokens = 10  # number of tokens generated in each sample
temperature = 0.8  # higher temperature (up to 1) is more random, lower (down to 0) means more greedy
top_k = (
    200  # retain only the top_k most likely tokens, clamp others to have 0 probability
)
seed = 1337
model_type = "gpt2"  # 'gpt2' or 'gpt2-medium' or 'gpt2-large' or 'gpt2-xl'
# exec(open("configurator.py").read())  # overrides from command line or config file
# -----------------------------------------------------------------------------
# load tokenizer
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# model
override_args = dict(dropout=0.0)
module = GPT.from_pretrained(model_type, override_args)
# initialize cache on all CausalSelfAttention layers


# %%
start_ids = jnp.array(encode(start), dtype=jnp.int32)
total_tokens = len(start_ids) + max_new_tokens
# module.on_all(CausalSelfAttention, lambda m: m.init_cache(1, total_tokens))

token = start_ids[0]
tokens = []
for i in range(total_tokens):
    if i < len(start_ids):
        token = start_ids[i]

    tokens.append(token)
    step_key = jax.random.PRNGKey(seed + i)
    # ctx = nnx.context(flags=dict(deterministic=True, inference=True))
    # logits, _ = module(token[None, None], ctx=ctx)
    ctx = nnx.context(flags=dict(deterministic=True, inference=False))
    logits, _ = module(jnp.stack(tokens)[None], ctx=ctx)
    logits = logits[:, 0, :] / temperature
    top_logits, top_tokens = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))
    token_idx = jax.random.categorical(step_key, top_logits, axis=-1)
    next_token = jnp.take_along_axis(top_tokens, token_idx[:, None], axis=-1)
    token = next_token[0, 0]

    print(decode([int(t) for t in tokens]))

tokens.append(token)
print(decode([int(t) for t in tokens]))


# %%
###
model = module.partition()
del module


# encode the beginning of the prompt
x = start_ids[None]
key = jax.random.PRNGKey(seed)


@partial(jax.jit, donate_argnums=(0,))
@print_compiling
def _sample(model: nnx.PureModule[GPT], key, tokens):
    return model.apply.generate(
        key,
        tokens,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        temperature=temperature,
    )


def sample(model, key, tokens) -> Tuple[str, nnx.PureModule[GPT]]:
    tokens, model = _sample(model, key, tokens)
    tokens = np.asarray(tokens, dtype=np.int32)
    return decode(tokens[0]), model


# run generation
for k in range(num_samples):
    step_key = jax.random.fold_in(key, k)
    sample_str, model = sample(model, step_key, x)
    print(sample_str)
    print("---------------")
