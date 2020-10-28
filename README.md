![alt text](https://github.com/chrisby/torchMTL/blob/main/torchmtl_logo.png "torchMTL Logo")    
A lightweight module for Multi-Task Learning in pytorch.

`torchmtl` tries to help you composing modular multi-task architectures with minimal effort. All you need is a list of dictionaries in which you define your layers and how they build on each other. From this, `torchmtl` constructs a meta-computation graph which is executed in each forward pass of the created `MTLModel`. To combine outputs from multiple layers, simple [wrapper functions](https://github.com/chrisby/torchMTL/blob/main/torchmtl/wrapping_layers.py) are provided.

### Installation
`torchmtl` can be installed via `pip`:
```
pip install torchmtl
```

### Quickstart
Assume you want to use two different embeddings of your input, combine them and then solve different prediction tasks.
