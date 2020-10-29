![torchMTL Logo](https://github.com/chrisby/torchMTL/blob/main/images/torchmtl_logo.png "torchMTL Logo")    
A lightweight module for Multi-Task Learning in pytorch.

`torchmtl` tries to help you composing modular multi-task architectures with minimal effort. All you need is a list of dictionaries in which you define your layers and how they build on each other. From this, `torchmtl` constructs a meta-computation graph which is executed in each forward pass of the created `MTLModel`. To combine outputs from multiple layers, simple [wrapper functions](https://github.com/chrisby/torchMTL/blob/main/torchmtl/wrapping_layers.py) are provided.

### Installation
`torchmtl` can be installed via `pip`:
```
pip install torchmtl
```

### Quickstart
Assume you want to train a network on three tasks as shown below.  
![example](https://github.com/chrisby/torchMTL/blob/main/images/example.png "example")  

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

You can build your final model with the following lines in which you specify from which layers you would like to receive the output.
```python
from torchmtl import MTLModel
model = MTLModel(tasks, output_tasks=['Task1', 'Task2', 'Task3'])
```

This constructs a **meta-computation graph** which is executed in each forward pass of your `model`. You can verify whether the graph was properly built by plotting it using the `networkx` library:
```python
import networkx as nx
pos = nx.planar_layout(model.g)
nx.draw(model.g, pos, font_size=14, node_color="y", node_size=450, with_labels=True)
```
![graph example](https://github.com/chrisby/torchMTL/blob/main/images/torchmtl_graph.png "graph example")  

#### The training loop
You can now enter the typical `pytorch` training loop and you will have access to everything you need to update your model:
```python
for X, y in data_loader:
    optimizer.zero_grad()

    # Our model will return a list of predictions,
    # loss functions, and regularization parameters (as defined in the tasks variable)
    y_hat, l_funcs, l_weights = model(X)
    
    loss = None
    # We can now iterate over the tasks and accumulate the losses
    for i in range(len(y_hat)):
        if not loss:
            loss = l_weights[i] * l_funcs[i](y_hat[i], y[i])
        else:
            loss += l_weights[i] * l_funcs[i](y_hat[i], y[i])
    
    loss.backward()
    optimizer.step()

```

### Details on the layer definition
There are 6 keys that can be specified (`name` and `layers` **must** always be present).  
`layers`: basically takes any `nn.Module` that you can think of. You can plug in a `transformer` or just a handful of fully connected layers.  
`anchor_layer`: This defines from which other layer this layer receives its input. Take care that the respective dimensions match.  
`loss`: The loss function you want to compute on the output of this layer (`l_funcs`).  
`loss_weight`: The scalar with which you want to regularize the respective loss (`l_weights`). If set to `'auto'`, a `nn.Parameter` is returned which will be updated through backpropagation.  
`loss_init_val`: Only needed if `loss_weight='auto'`. The initialization value of the `loss_weight` parameter.

### Wrapping functions
Nodes of the **meta-computation graph** don't have to be pytorch Modules. They can be *concatenation* functions or indexing functions that return a certain element of the input. If your `X` consists of two types of input data `X=[X_1, X_2]`, you can use the `SimpleSelect` layer to select the `X_1` by setting  
```python
from torchmtl.wrapping_layers import SimpleSelect
{ ...,
  'layers' = SimpleSelect(seleciton_axis=0),
  ...
}
```
It should be trivial to write your own wrapping layers, but I try to provide useful ones with this library. If you have any layers in mind but no time to implement them, feel free to [open an issue](https://github.com/chrisby/torchMTL/issues).

Logo credits and license: I reused and remixed (moved the dot and rotated the resulting logo a couple times) the pytorch logo from [here](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png) (accessed through [wikimedia commons](https://commons.wikimedia.org/wiki/File:Pytorch_logo.png)) which can be used under the [Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/deed.en) license. Hence, this logo falls under the same license. 
