import torch
from torch_persistent_homology.persistent_homology_cpu import uf_find, uf_merge


def test_union_find():
    vertices = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7]).int()
    parents = vertices.clone()
    for v in vertices:
        assert uf_find(parents, v) == v
    uf_merge(parents, 0, 1)
    uf_merge(parents, 4, 5)
    uf_merge(parents, 4, 7)
    assert uf_find(parents, 0) == uf_find(parents, 1)
    assert uf_find(parents, 4) == uf_find(parents, 5)
    assert uf_find(parents, 5) == uf_find(parents, 4)
    assert uf_find(parents, 4) == uf_find(parents, 7)
    assert uf_find(parents, 7) == uf_find(parents, 4)

    uf_merge(parents, 2, 3)
    uf_merge(parents, 0, 4)
    assert uf_find(parents, 2) == uf_find(parents, 3)
    assert uf_find(parents, 6) == 6
