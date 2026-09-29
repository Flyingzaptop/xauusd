"""Instrument neural scheduling with exact additive cells, independently of GRUs."""
import numpy as np
import torch
from torch import nn

from src.crossing_memory_v2 import TreeCrossingMemory, route_windows_v2


class AddObservation(nn.Module):
    def forward(self, inputs, hidden):
        return hidden + inputs[:, :1]


class AddChild(nn.Module):
    def forward(self, inputs, hidden):
        return hidden + inputs[:, :1]


def test_every_observation_has_one_state_owner_through_cascades():
    paths=np.array([[0,6,9,10,-2,-8,-5,-2], [0,6,3,2,0,6,9,10]],dtype=float)
    routed={k:torch.from_numpy(v) for k,v in route_windows_v2(paths,[6,3,1]).items()}
    routed['metadata']=routed['metadata'].double()
    model=TreeCrossingMemory(1,1,4).double()
    model.observe=nn.ModuleList([AddObservation() for _ in range(4)])
    model.merge=nn.ModuleList([AddChild() for _ in range(3)])
    x=torch.tensor([[[2.**t] for t in range(8)]]*2,dtype=torch.float64,requires_grad=True)
    state=None
    for t in range(8):
        _,state=model.step(x[:,t],{k:v[:,t] for k,v in routed.items()},state)
        torch.testing.assert_close(state.sum((1,2)),x[:,:t+1].sum((1,2)),rtol=0,atol=0)
        assert torch.all(state[~routed['active'][:,t]]==0)
    state.sum().backward()
    torch.testing.assert_close(x.grad,torch.ones_like(x),rtol=0,atol=0)
