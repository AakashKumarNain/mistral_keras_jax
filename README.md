# Mistral-7B reference implementation in Keras and JAX

This repository contains a port of the original [Mistral-7B model](https://github.com/mistralai/mistral-src/tree/main?tab=readme-ov-file) in Keras and JAX. The model here isn't pretrained or fine-tuned. The weights are ported from torch to Keras, provided on an "as is" basis, without warranties or conditions of any kind.

Any official restriction, if applicable, that comes with the original code and the model, applies here as well. Please check the original license and the [repo](https://github.com/mistralai/mistral-src/tree/main?tab=readme-ov-file) for the details.


## Tasks

- [x] Build the model with a 1:1 mapping from PyTorch to Keras
- [x] Port weights from Torch to Keras
- [x] Test with different dtypes
- [x] Provide a single script to run end-to-end generation
- [ ] Sharded inference
- [ ] HLO graphs
- [ ] Keras model optimizations

## Instructions

1. Clone the repo.
2. Run the `generate.py` file located inside `src` to run generation for sample example.


## Call for Contributions

You can pick any task from the above tasks list. Please open an issue regarding the feature you are working on.
Also, we use `ruff` as the linter and formatter. Please format your code using the shell scripts provided before
sending a PR.

**Note:** Any nonsensical PR like updating the readme or adding docs to the existing functions will not be accepted
at this point.


## References

[1] [Mistral 7B- Official code implementation](https://github.com/mistralai/mistral-src/tree/main?tab=readme-ov-file)

[2] [Generating Long Sequences with Sparse Transformers, Child et al. 2019](https://arxiv.org/pdf/1904.10509.pdf)

[3] [Longformer: The Long-Document Transformer, Beltagy et al. 2020](https://arxiv.org/pdf/2004.05150v2.pdf)
