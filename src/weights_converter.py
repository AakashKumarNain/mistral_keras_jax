import gc
import pickle
import warnings
from keras import ops
from pathlib import Path


# Note: Using pickle because I am just being lazy. There are better alternatives!
def save_torch_weights_as_numpy_arrays(state_dict, save_path):
    """Convert torch weights to numpy and save them as `ndarrays`.

    Args:
        state_dict: Ordered dict containing the weights of Mistral-7B
            model represented in fp16 format.
        save_path: Path to save the pickle file.
    """

    if state_dict is None:
        raise ValueError(f"Expected an OrderedDict. Received {state_dict}.")

    save_path = Path(save_path)
    if save_path.suffix not in (".pkl", ".pickle"):
        raise ValueError(
            "Expected `save_path` file to have `.pkl` or `pickle` extension."
        )

    new_state_dict = {}

    try:
        for key, value in state_dict.items():
            if "embeddings" not in key and "norm" not in key:
                # The weights of the Dense layers in Keras are transposed
                new_state_dict[key] = state_dict[key].cpu().numpy().T
            else:
                new_state_dict[key] = state_dict[key].cpu().numpy()

        with open(save_path, "wb") as f:
            pickle.dump(new_state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    except Exception as ex:
        print("Oh oh! Something went wrong!\n", ex)

    del new_state_dict
    gc.collect()


def load_numpy_weights_to_keras(numpy_weights_file, model):
    """Load numpy weights to keras model.

    **Note:** To load the weights, you need to be build the model first.
    You can build the model by passing fake inputs to the model. For example:

    ```python
    >>> cache_k = ops.zeros((max_batch_size, n_layers, sliding_window, n_kv_heads, head_dim), dtype="float16")
    >>> cache_v = ops.zeros((max_batch_size, n_layers, sliding_window, n_kv_heads, head_dim), dtype="float16")
    >>> fake_pos = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
    >>> fake_inp = jnp.asarray([[1,  832,  349,  265, 1369]], dtype=jnp.int32)
    >>>  _ = model([fake_inp, fake_pos[None, :], cache_k, cache_v])
    ```

    Args:
        state_dict: Ordered dict containing the weights of Mistral-7B
            model represented in fp16 format.
        save_path: Path to save the pickle file.

    Returns:
        model with weights loaded from the pickle file
    """

    file_path = Path(numpy_weights_file)

    if file_path.suffix not in (".pkl", ".pickle"):
        raise ValueError(
            "Expected `numpy_weights_file` file to have `.pkl` or `pickle` extension."
        )

    if not model.built:
        warnings.warn("The model has't been built yet. Please build the model first.")
        return

    try:
        with open(file_path, "rb") as f:
            state_dict = pickle.load(f)

        for key, variable in zip(state_dict.keys(), model.trainable_variables):
            weights = ops.convert_to_tensor(state_dict[key], dtype="float16")
            variable.assign(weights)
            del weights
            gc.collect()

        del state_dict
        gc.collect()

    except Exception as ex:
        print("Oh oh! Something went wrong!\n", ex)

    return model
