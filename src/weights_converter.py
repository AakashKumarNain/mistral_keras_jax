import pickle
import torch
import numpy as np
from keras import ops
from pathlib import Path


# Note: Using pickle because I am just being lazy. There are better alternatives!
def save_torch_weights_as_numpy_arrays(state_dict, save_path):
    """"Convert torch weights to numpy and save them as `ndarrays`.

    Args:
        state_dict: Ordered dict containing the weights of Mistral-7B
            model represented in fp16 format.
        save_path: Path to save the pickle file.
    """

    if state_dict is None:
        raise ValueError(f"Expected an OrderedDict. Received {state_dict}.")
    
    save_path = Path(save_path)
    if save_path.suffix != ".pkl":
        raise ValueError(f"Expected `save_path` to have `.pkl` extension")

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
    
    except Exception as e:
        print("Oh oh! Something went wrong!\n", e)




def convert_numpy_weights
