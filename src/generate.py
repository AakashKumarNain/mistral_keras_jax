import os

os.environ["KERAS_BACKEND"] = "jax"

import json
import numpy as np
from typing import NamedTuple

import jax
import jax.numpy as jnp

import keras
from keras import ops

from model import Transformer
from tokenizer import Tokenizer

DTYPE_POLICY = "float16"
keras.config.set_dtype_policy(DTYPE_POLICY)


def generate(prompts, model, tokenizer, cache_k, cache_v, max_tokens=30):
    """Given a list of prompts, outputs `n` tokens.

    Args:
        prompts: List of strings
        model: The Mistral-7B model
        tokenizer: The Mistral tokenizer
        cache_k, cache_v: kv-cache for all attention layers

    Returns:
        A list of strings containing generated outputs
    """

    # 1. Encode the prompts
    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]
    prompt_lens = [len(x) for x in encoded_prompts]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)

    # 2. Using numpy to generate the desired input. Will replace it with something
    # better later on
    input_tokens = np.full(
        (len(prompts), max_prompt_len), tokenizer.pad_id, dtype=np.int32
    )
    for i, encoded in enumerate(encoded_prompts):
        input_tokens[i, : len(encoded)] = ops.convert_to_tensor(encoded)
    input_mask = input_tokens != tokenizer.pad_id

    # 3. pre-fill
    positions = jnp.arange(0, min_prompt_len)
    logits, cache_k, cache_v = model(
        [jnp.asarray(input_tokens[:, :min_prompt_len]), positions, cache_k, cache_v]
    )
    logprobs = jax.nn.log_softmax(ops.cast(logits, "float32"), axis=-1)

    # 4. Decode
    generated = []
    all_logprobs = [
        ops.squeeze(
            ops.take_along_axis(
                logprobs[:, :-1, :], input_tokens[:, 1:min_prompt_len, None], axis=2
            ),
            axis=-1,
        )
    ]
    cur_pos = min_prompt_len

    for _ in range(max_tokens):
        next_token = ops.argmax(logprobs[:, -1, :], axis=-1)

        if cur_pos < input_mask.shape[1]:
            next_token = ops.where(
                input_mask[:, cur_pos], input_tokens[:, cur_pos], next_token
            )

        all_logprobs.append(
            ops.take_along_axis(logprobs[:, -1, :], next_token[:, None], axis=1)
        )

        generated.append(next_token[:, None])

        logits, cache_k, cache_v = model(
            [next_token[:, None], jnp.array([cur_pos]), cache_k, cache_v]
        )
        logprobs = jax.nn.log_softmax(ops.cast(logits, "float32"), axis=-1)
        cur_pos += 1

    # 5. Convert to strings
    all_logprobs = ops.concatenate(all_logprobs, axis=1)
    res = []
    if max_tokens > 0:
        generated = ops.concatenate(generated, axis=1)

        for i, x in enumerate(encoded_prompts):
            next_pred_token = tokenizer.decode(
                x[:min_prompt_len] + generated[i].tolist()
            )
            res.append(next_pred_token)
    return res, all_logprobs


class ModelArgs(NamedTuple):
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_dim: int
    vocab_size: int
    sliding_window: int
    norm_eps: float
    max_batch_size: int = 1


def main():
    # 1. Read the config for Mistral-7B
    with open("../model_files/model_args.json", "r") as f:
        args = ModelArgs(**json.loads(f.read()))

    # 2. Load the tokenizer
    tokenizer = Tokenizer("../../mistral-7B-v0.1/tokenizer.model")

    # 3. Build the model
    model = Transformer(args)

    # 4. kv cache tensors. Each attention layer has a cache associated with it
    cache_k = ops.zeros(
        (
            args.n_layers,
            args.max_batch_size,
            args.sliding_window,
            args.n_kv_heads,
            args.head_dim,
        ),
        dtype="float16",
    )
    cache_v = ops.zeros(
        (
            args.n_layers,
            args.max_batch_size,
            args.sliding_window,
            args.n_kv_heads,
            args.head_dim,
        ),
        dtype="float16",
    )

    # 5. Sample inputs
    prompts = ["This is a test"]  # Using only one input as our `max_batch_`size is 1

    # 6. Generate
    results, logprobs = generate(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        cache_k=cache_k,
        cache_v=cache_v,
    )

    # 7. Check the output
    for inp, out in zip(prompts, results):
        print("Inputs:")
        print(inp)
        print("Outputs:")
        print(out)
        print("=" * 75)


if __name__ == "__main__":
    main()
