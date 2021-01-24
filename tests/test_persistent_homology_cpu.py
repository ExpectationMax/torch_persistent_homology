import torch
from torch_persistent_homology import persistent_homology_cpu


def test_set_to_one():
    z = torch.zeros(50)
    persistent_homology_cpu.set_to_one(z)
    assert (z == 1.).all()


def test_ones_tensor():
    z = persistent_homology_cpu.ones_tensor()
    assert (z == 1.).all()
