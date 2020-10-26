"""Module with functions to construct a holistic torch.nn.Model from a set of
models."""
import logging

from torch import nn
import networkx as nx
from copy import copy # only used in logging.DEBUG mode

LAYER_KEY = 'layers'
NAME_KEY = 'name'
ANCHOR_KEY = 'anchor_layer'

class MTLModel(nn.Module):
    """
    A torch.nn.Module built from a set of shared and task specific layers

    Attributes
    ----------
    name : type 
        Description

    Methods
    -------
    bla()
        Description

    """

    def __init__(self, task_layers, output_tasks):
        super(MTLModel, self).__init__()
        self.task_layers = task_layers
        self.output_tasks = output_tasks
        self.layer_names = [t[NAME_KEY] for t in task_layers]

        self.g = nx.DiGraph()
        self.g.add_node('root')
        self._build_graph()

    def _bfs_forward(self, start_node):
        ''' Here we iteratore through the graph in a BFS-fashion starting from
        `start_node`, typically this is the `root` node. This node is skipped
        and we pass the input data and resulting outputs from all layers foward.
        '''
        visited = {node: False for node in self.layer_names}

        # First node is visited
        queue = [start_node]
        visited[start_node] = True

        while queue:
            node = queue.pop(0)
            if node != start_node:
                input_nodes = self.g.predecessors(node)
                if logging.getLogger().level == logging.DEBUG:
                    l = copy(input_nodes)
                    print(f"Feeding output from {list(l)} into {node}")
                cur_layer = getattr(self, node)
                output_pre_layers = [self.outputs[n] for n in input_nodes]
                cur_output = cur_layer(*output_pre_layers)

                if node not in self.outputs.keys():
                    self.outputs[node] = cur_output

            for i in self.g.successors(node):
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True

        return [self.outputs[t] for t in self.output_tasks]

    def forward(self, input):
        self.outputs = {'root': input}
        return self._bfs_forward('root')

    def _build_graph(self):
        for layer in self.task_layers:
            self._add_layer(layer)
            self._add_to_graph(layer)

    def _add_to_graph(self, layer):
        layer_name = layer[NAME_KEY]
        self._add_node(layer_name)

        if 'anchor_layer' not in layer.keys():
            # If there is no anchor layer, we expect it to be a layer which
            # receives data inputs and is hence connected to the root node
            self.g.add_edge('root', layer_name)
        else:
            anchor_layer = layer[ANCHOR_KEY]
            if isinstance(anchor_layer, list):
                for a_l_name in anchor_layer:
                    self._add_node(a_l_name)
                    self.g.add_edge(a_l_name, layer_name)
            else:
                self._add_node(anchor_layer)
                self.g.add_edge(anchor_layer, layer_name)

    def _add_node(self, layer):
        if isinstance(layer, str):
            layer_name = layer
            self.g.add_node(layer_name)
        else:
            layer_name = layer[NAME_KEY]
            self.g.add_node(layer_name)
            if 'anchor_layer' not in layer.keys():
                self.g.add_edge('root', layer_name)
    
    def _add_layer(self, layer):
        layer_modules = layer[LAYER_KEY]
        layer_name_main = layer[NAME_KEY]
        setattr(self, layer_name_main, layer_modules)
