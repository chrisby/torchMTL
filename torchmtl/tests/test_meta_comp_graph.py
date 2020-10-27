import logging

import pytest
import networkx as nx
from networkx.algorithms import is_isomorphic

import torch
from torch import nn
from torch import optim
from torch.nn import (Sequential, Linear, MSELoss)

from torchmtl.model_builder import MTLModel
from torchmtl.wrapping_layers import Concat

@pytest.fixture
def complex_tasks():
    return [
        {
            'name': "InputTask2",
            'layers': Sequential(*[Linear(16, 32), Linear(32, 8)]),
            'loss': MSELoss(),
            'loss_weight': 1.0,
            # No anchor_layer means this layer receives input directly
        },    
        {
            'name': "InputTask1",
            'layers': Sequential(*[Linear(16, 32), Linear(32, 8)]),
            'loss_weight': 1.0
            # No anchor_layer means this layer receives input directly
        },
        {
            'name': "CombTask",
            'layers': Concat(dim=1),
            'loss_weight': 1.0,
            'anchor_layer': ['InputTask1', 'InputTask2']
        },
        {
            'name': "AuxTask1",
            'layers': Sequential(*[Linear(16, 32), Linear(32, 1)]),
            'loss': MSELoss(),
            'loss_weight': 1.0,
            'anchor_layer': 'CombTask'
        },
        {
            'name': "MiddleTask1",
            'layers': Sequential(*[Linear(16, 32), Linear(32, 8)]),
            'loss': MSELoss(),
            'loss_weight': 1.0,
            'anchor_layer': 'CombTask'
        },    
        {
            'name': "AuxTask2",
            'layers': Sequential(*[Linear(8, 32), Linear(32, 1)]),
            'loss': MSELoss(),
            'loss_weight': 1.0,
            'anchor_layer': 'MiddleTask1'
        },
        {
            'name': "MiddleTask2",
            'layers': Sequential(*[Linear(8, 32), Linear(32, 4)]),
            'loss': MSELoss(),
            'loss_weight': 1.0,
            'anchor_layer': 'MiddleTask1'
        },
        {
            'name': "AuxTask3",
            'layers': Sequential(*[Linear(4, 32), Linear(32, 1)]),
            'loss': MSELoss(),
            'loss_weight': 1.0,
            'anchor_layer': 'MiddleTask2'
        }
    ]

class TestGraphGeneration:
    def test_complex_generation(self, complex_tasks):

        ground_truth_graph = nx.DiGraph()
        ground_truth_graph.add_nodes_from(['root', 'InputTask1', 'InputTask2',
                                           'CombTask', 'AuxTask1',
                                           'MiddleTask1', 'AuxTask2',
                                           'MiddleTask2', 'AuxTask3'])
        ground_truth_graph.add_edges_from([('root', 'InputTask1'),
                                           ('root', 'InputTask2'),
                                           ('InputTask1', 'CombTask'),
                                           ('InputTask2', 'CombTask'),
                                           ('CombTask', 'AuxTask1'),
                                           ('CombTask', 'MiddleTask1'),
                                           ('MiddleTask1', 'AuxTask2'),
                                           ('MiddleTask1', 'MiddleTask2'),
                                           ('MiddleTask2', 'AuxTask3')])
        model = MTLModel(complex_tasks, output_tasks=['AuxTask1', 'AuxTask2', 'AuxTask3'])
        assert is_isomorphic(model.g, ground_truth_graph)

class TestGraphExecution:
    def test_3_mse_tasks(self, complex_tasks):
        model = MTLModel(complex_tasks, output_tasks=['AuxTask1', 'AuxTask2', 'AuxTask3'])

        sample_size = 16
        torch.manual_seed(0)
        X = torch.rand((sample_size, 16))
        truths_1 = torch.ones(sample_size, 1) * 10.
        truths_2 = torch.ones(sample_size, 1) * 20.
        truths_3 = torch.ones(sample_size, 1) * 30.

        mse = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.01)
        num_it = 1000

        logging.getLogger().warning(f"Training 3 MSE losses for {num_it} iterations")
        for i in range(num_it):
            optimizer.zero_grad()

            preds = model(X)

            loss_1 = mse(truths_1, preds[0])
            loss_2 = mse(truths_2, preds[1])
            loss_3 = mse(truths_3, preds[2])

            loss = loss_1 + loss_2 + loss_3
            loss.backward()
            optimizer.step()
       
        assert torch.isclose(preds[0], truths_1, atol=1.0).all()
        assert torch.isclose(preds[1], truths_2, atol=1.0).all()
        assert torch.isclose(preds[2], truths_3, atol=1.0).all()


    def test_3_mse_tasks_2_losses(self, complex_tasks):
        model = MTLModel(complex_tasks, output_tasks=['AuxTask1', 'AuxTask2', 'AuxTask3'])

        sample_size = 16
        torch.manual_seed(0)
        X = torch.rand((sample_size, 16))
        truths_1 = torch.ones(sample_size, 1) * 10.
        truths_2 = torch.ones(sample_size, 1) * 20.
        truths_3 = torch.ones(sample_size, 1) * 30.

        mse = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.01)
        num_it = 1000

        logging.getLogger().warning(f"Training 3 MSE losses for {num_it} iterations")
        for i in range(num_it):
            optimizer.zero_grad()

            preds = model(X)

            loss_1 = mse(truths_1, preds[0])
            loss_3 = mse(truths_3, preds[2])

            loss = loss_1 + loss_3
            loss.backward()
            optimizer.step()
       
        assert torch.isclose(preds[0], truths_1, atol=1.0).all()
        assert not torch.isclose(preds[1], truths_2, atol=1.0).all()
        assert torch.isclose(preds[2], truths_3, atol=1.0).all()


