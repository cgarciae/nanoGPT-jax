"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import nnx
from flax.training import train_state


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1


class CausalSelfAttention(nnx.Module):
    def __init__(self, config: GPTConfig, *, ctx: nnx.Context):
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nnx.Linear(config.n_embd, 3 * config.n_embd, ctx=ctx)
        # output projection
        self.c_proj = nnx.Linear(config.n_embd, config.n_embd, ctx=ctx)
        # regularization
        self.attn_dropout = nnx.Dropout(config.dropout)
        self.resid_dropout = nnx.Dropout(config.dropout)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def __call__(self, x: jax.Array, *, ctx: nnx.Context) -> jax.Array:
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = x.shape

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = jnp.split(self.c_attn(x), 3, axis=-1)
        # (B, nh, T, hs)
        q = q.reshape(B, T, self.n_head, C // self.n_head).swapaxes(1, 2)
        # (B, nh, T, hs)
        k = k.reshape(B, T, self.n_head, C // self.n_head).swapaxes(1, 2)
        # (B, nh, T, hs)
        v = v.reshape(B, T, self.n_head, C // self.n_head).swapaxes(1, 2)

        mask = jnp.tril(jnp.ones((T, T))).reshape((1, 1, T, T))

        # causal self-attention; Self-attend:
        #   (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.swapaxes(-2, -1)) * (1.0 / jnp.sqrt(k.shape[-1]))
        att = jnp.where(mask == 0, float("-inf"), att)
        att = nnx.softmax(att, axis=-1)
        att = self.attn_dropout(att, ctx=ctx)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.swapaxes(1, 2).reshape(B, T, C)
        # output projection
        y = self.resid_dropout(self.c_proj(y), ctx=ctx)
        return y


class MLP(nnx.Module):
    def __init__(self, config: GPTConfig, *, ctx: nnx.Context):
        self.c_fc = nnx.Linear(config.n_embd, 4 * config.n_embd, ctx=ctx)
        self.c_proj = nnx.Linear(4 * config.n_embd, config.n_embd, ctx=ctx)
        self.dropout = nnx.Dropout(config.dropout)

    def __call__(self, x: jax.Array, *, ctx: nnx.Context) -> jax.Array:
        x = self.c_fc(x)
        x = nnx.gelu(x, approximate=True)
        x = self.c_proj(x)
        x = self.dropout(x, ctx=ctx)
        return x


class Block(nnx.Module):
    def __init__(self, config: GPTConfig, *, ctx: nnx.Context):
        self.ln_1 = nnx.LayerNorm(config.n_embd, epsilon=1e-5, ctx=ctx)
        self.attn = CausalSelfAttention(config, ctx=ctx)
        self.ln_2 = nnx.LayerNorm(config.n_embd, epsilon=1e-5, ctx=ctx)
        self.mlp = MLP(config, ctx=ctx)

    def __call__(self, x: jax.Array, *, ctx: nnx.Context) -> jax.Array:
        x = x + self.attn(self.ln_1(x), ctx=ctx)
        x = x + self.mlp(self.ln_2(x), ctx=ctx)
        return x


class GPT(nnx.Module):
    def __init__(self, config: GPTConfig, *, ctx: nnx.Context):
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.wte = nnx.Embed(config.vocab_size, config.n_embd, ctx=ctx)
        self.wpe = nnx.Embed(config.block_size, config.n_embd, ctx=ctx)
        self.drop = nnx.Dropout(config.dropout)
        self.h = nnx.Sequence(Block(config, ctx=ctx) for _ in range(config.n_layer))
        self.ln_f = nnx.LayerNorm(config.n_embd, epsilon=1e-5, ctx=ctx)

        self.config = config

    def __call__(
        self, idx: jax.Array, *, ctx: nnx.Context, targets: Optional[jax.Array] = None
    ):
        b, t = idx.shape
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = jnp.arange(0, t, dtype=jnp.int32)[None]  # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = self.drop(tok_emb + pos_emb, ctx=ctx)
        for block in self.h:
            x = block(x, ctx=ctx)
        x = self.ln_f(x)

        logits = self.wte.attend(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, targets
            ).mean()
        else:
            loss = None

        return logits, loss

    def crop_block_size(self, params, block_size: int):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model

        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.wpe.embedding = self.wpe.embedding[:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        # we can override the dropout rate
        if "dropout" in override_args:
            config_args["dropout"] = override_args["dropout"]
        # block_size is always 1024 for GPT model checkpoints
        # if one wants a lower block_size it has to be done through model surgery
        # later, by calling crop_block_shape

        # create a from-scratch initialized minGPT model
        config = GPTConfig(block_size=1024, **config_args)
        ctx = nnx.Context(jax.random.PRNGKey(0), flags=dict(deterministic=False))
        model = GPT(config, ctx=ctx)
        ref_dict = model.mutable_state_dict(".")

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        def copy_from(flax_name, pt_name):
            pt_tensor = sd_hf[pt_name]
            leaf = ref_dict[flax_name]
            pt_array = pt_tensor.detach().cpu().numpy()

            assert pt_array.shape == leaf.value.shape

            leaf.value = pt_array

        # transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        copy_from("wte.embedding", "transformer.wte.weight")
        copy_from("wpe.embedding", "transformer.wpe.weight")

        for i in range(config.n_layer):
            copy_from(f"h.{i}.ln_1.scale", f"transformer.h.{i}.ln_1.weight")
            copy_from(f"h.{i}.ln_1.bias", f"transformer.h.{i}.ln_1.bias")
            copy_from(
                f"h.{i}.attn.c_attn.kernel", f"transformer.h.{i}.attn.c_attn.weight"
            )
            copy_from(f"h.{i}.attn.c_attn.bias", f"transformer.h.{i}.attn.c_attn.bias")
            copy_from(
                f"h.{i}.attn.c_proj.kernel", f"transformer.h.{i}.attn.c_proj.weight"
            )
            copy_from(f"h.{i}.attn.c_proj.bias", f"transformer.h.{i}.attn.c_proj.bias")
            copy_from(f"h.{i}.ln_2.scale", f"transformer.h.{i}.ln_2.weight")
            copy_from(f"h.{i}.ln_2.bias", f"transformer.h.{i}.ln_2.bias")
            copy_from(f"h.{i}.mlp.c_fc.kernel", f"transformer.h.{i}.mlp.c_fc.weight")
            copy_from(f"h.{i}.mlp.c_fc.bias", f"transformer.h.{i}.mlp.c_fc.bias")
            copy_from(
                f"h.{i}.mlp.c_proj.kernel", f"transformer.h.{i}.mlp.c_proj.weight"
            )
            copy_from(f"h.{i}.mlp.c_proj.bias", f"transformer.h.{i}.mlp.c_proj.bias")

        copy_from("ln_f.scale", "transformer.ln_f.weight")
        copy_from("ln_f.bias", "transformer.ln_f.bias")

        return model

    @staticmethod
    def configure_optimizers(params: nnx.State, weight_decay, learning_rate, betas):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        def get_optimizer(decay):
            return optax.adamw(
                learning_rate=learning_rate,
                b1=betas[0],
                b2=betas[1],
                weight_decay=decay,
            )

        def partition_fn(path: Tuple[str, ...], x) -> str:
            if path[-1] in ("bias", "scale", "embedding"):
                return "no_decay"
            elif path[-1] in ("kernel",):
                return "decay"
            else:
                raise ValueError(f"Unrecognized parameter: {path}")

        partition_optimizers = {
            "decay": get_optimizer(weight_decay),
            "no_decay": get_optimizer(0.0),
        }
        param_partitions = nnx.State(
            (path, partition_fn(path, x)) for path, x in params.items()
        )
        tx = optax.multi_transform(partition_optimizers, param_partitions)

        return tx

    # @torch.no_grad()
    def generate(self, key, input_tokens, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        B, T = input_tokens.shape
        padding = jnp.zeros((B, max_new_tokens), dtype=jnp.int32)
        tokens = jnp.concatenate([input_tokens, padding], axis=-1)
        indexes = jnp.arange(T, T + max_new_tokens)

        # tokens index -> tokens None
        def scan_f(tokens, i):
            # l: x y
            # t: a b - -
            # i: 0 1 2 3
            step_key = jax.random.fold_in(key, i)
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            ctx = nnx.Context(flags=dict(deterministic=True))
            logits, _ = self(tokens, ctx=ctx)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, i - 1, :] / temperature
            # optionally crop the logits to only the top k options
            # sample from the distribution
            if top_k is not None:
                top_logits, top_tokens = jax.lax.top_k(
                    logits, min(top_k, logits.shape[-1])
                )
                token_idx = jax.random.categorical(step_key, top_logits, axis=-1)
                next_token = jnp.take_along_axis(
                    top_tokens, token_idx[:, None], axis=-1
                ).squeeze(-1)
            else:
                next_token = jax.random.categorical(step_key, logits, axis=-1)
                # logits = jnp.where(logits < v[:, -1:], float('-inf'), logits)
            # append sampled index to the running sequence and continue
            tokens = tokens.at[:, i].set(next_token)

            return tokens, None

        tokens, _ = jax.lax.scan(scan_f, tokens, indexes)

        return tokens

    def create_state(
        self,
        learning_rate,
        weight_decay,
        beta1,
        beta2,
        decay_lr=None,
        warmup_iters=None,
        lr_decay_iters=None,
        min_lr=None,
        params=None,
        **kwargs,
    ):
        _params, moduledef = self.partition("params")
        if params is None:
            params = _params

        if decay_lr:
            assert (
                warmup_iters is not None
                and lr_decay_iters is not None
                and min_lr is not None
            )
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_iters,
                decay_steps=lr_decay_iters,
                end_value=min_lr,
            )
        else:
            lr_schedule = learning_rate
        tx = self.configure_optimizers(
            params,
            weight_decay=weight_decay,
            learning_rate=lr_schedule,
            betas=(beta1, beta2),
        )
        return train_state.TrainState.create(
            apply_fn=moduledef.apply, params=params, tx=tx
        )
