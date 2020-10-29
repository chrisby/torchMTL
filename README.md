![torchMTL Logo](https://github.com/chrisby/torchMTL/blob/main/torchmtl_logo.png "torchMTL Logo")    
A lightweight module for Multi-Task Learning in pytorch.

`torchmtl` tries to help you composing modular multi-task architectures with minimal effort. All you need is a list of dictionaries in which you define your layers and how they build on each other. From this, `torchmtl` constructs a meta-computation graph which is executed in each forward pass of the created `MTLModel`. To combine outputs from multiple layers, simple [wrapper functions](https://github.com/chrisby/torchMTL/blob/main/torchmtl/wrapping_layers.py) are provided.

### Installation
`torchmtl` can be installed via `pip`:
```
pip install torchmtl
```

### Quickstart
Assume you want to train a network on three tasks as shown below.  
![alt text](https://github.com/chrisby/torchMTL/blob/main/example.png "example")  

To construct such an architecture with `torchmtl`, you simply have to define the following list

```python
tasks = [
        {
            'name': "Embed1",
            'layers': Sequential(*[Linear(16, 32), Linear(32, 8)]),
            # No anchor_layer means this layer receives input directly
        },    
        {
            'name': "Embed2",
            'layers': Sequential(*[Linear(16, 32), Linear(32, 8)]),
            # No anchor_layer means this layer receives input directly
        },
        {
            'name': "CatTask",
            'layers': Concat(dim=1),
            'loss_weight': 1.0,
            'anchor_layer': ['Embed1', 'Embed2']
        },
        {
            'name': "Task1",
            'layers': Sequential(*[Linear(8, 32), Linear(32, 1)]),
            'loss': MSELoss(),
            'loss_weight': 1.0,
            'anchor_layer': 'Embed1'            
        },
        {
            'name': "Task2",
            'layers': Sequential(*[Linear(8, 64), Linear(64, 1)]),
            'loss': BCEWithLogitsLoss(),
            'loss_weight': 1.0,
            'anchor_layer': 'Embed2'            
        }, 
        {
            'name': "FNN",
            'layers': Sequential(*[Linear(16, 32), Linear(32, 32)]),
            'anchor_layer': 'CatTask'
        },
        {
            'name': "Task3",
            'layers': Sequential(*[Linear(32, 16), Linear(16, 1)]),
            'anchor_layer': 'FNN',
            'loss': MSELoss(),
            'loss_weight': 'auto',
            'loss_init_val': 1.0
        }
    ]
```
