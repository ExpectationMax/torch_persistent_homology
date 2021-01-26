import torch
from torch_persistent_homology import persistent_homology_cpu
from pyper.utilities import UnionFind


def persistence_routine(filtered_v_, edge_indices, cycles=False):
    """
    Pytorch based routine to compute the persistence pairs
    Based on pyper routine.
    Inputs :
        * filtration values of the vertices
        * data object that stores the graph structure (could be just the edge_index actually)
        * method is just a check for the algo
        * cycles is a boolean to compute the 1D persistence or not. If true, returns also the 1D persistence.
    """

    # Quick check for the filtration values to be different.
    # if torch.unique(filtered_v_).reshape(-1,1).shape[0] != filtered_v_.reshape(-1,1).shape[0]:
    # if not unique, we add a small perturbation on all the values with std 0.01 x the initial std of the filtration values.
    # std = torch.std(filtered_v_)
    # filtered_v_ += 0.001*std*torch.randn(filtered_v_.shape)

    # Compute the edge filtrations as the max between the value of the nodes.

    filtered_e_, _ = torch.max(torch.stack(
        (filtered_v_[edge_indices[0]], filtered_v_[edge_indices[1]])), axis=0)

    # Only the edges need to be sorted, since they determine the
    # filtration ordering. For the vertices, we will look up the
    # values directly in the tensor.
    filtered_e, e_indices = torch.sort(filtered_e_)

    n_vertices = len(filtered_v_)

    uf = UnionFind(n_vertices)

    persistence = torch.zeros(
        (n_vertices, 2),
        device=filtered_v_.device
    )
    if cycles:
        persistence1 = torch.zeros(
            (len(filtered_e), 2), device=filtered_v_.device)

    edge_indices_cycles = []

    unpaired_value = filtered_e[-1]

    persistence[:, 0] = filtered_v_

    for edge_index, edge_weight in zip(e_indices, filtered_e):

        # nodes connected to this edge
        nodes = edge_indices[:, edge_index]

        younger = uf.find(nodes[0])
        older = uf.find(nodes[1])

        if younger == older:
            if cycles:
                # edge_indices_cycles.append(edge_index)
                persistence1[edge_index, 0] = filtered_e_[edge_index]
                persistence1[edge_index, 1] = unpaired_value
            continue
        else:
            # Use vertex weight lookup to determine which vertex comes
            # first. This works because our filtrations are based on
            # values at the vertices themselves.
            if filtered_v_[younger] < filtered_v_[older]:
                younger, older = older, younger
                nodes = torch.flip(nodes, [0])

        # persistence[younger, 0] = filtered_v_[younger]
        persistence[younger, 1] = edge_weight

        uf.merge(nodes[0], nodes[1])

    # TODO : this currently assumes a single unpaired value for the whole batch. THis can be discussed.
    for root in uf.roots():
        persistence[root, 0] = filtered_v_[root]
        persistence[root, 1] = unpaired_value

    # if cycles:
    # persistence1 = torch.zeros((len(filtered_e),2), device = filtered_v_.device)
    # for edge_index in edge_indices_cycles:
    #    persistence1[edge_index,0] = filtered_e_[edge_index]
    #    persistence1[edge_index,1] = unpaired_value

    # cycle_time = time.time()

    if cycles:
        return persistence, persistence1
    else:
        return persistence


def test_persistence_computation():
    edge_index = torch.Tensor([[0, 0, 2, 3, 4], [2, 3, 3, 4, 1]]).long()
    filtered_v = torch.Tensor([1., 1., 2., 3., 4.])
    filtered_e, _ = torch.max(torch.stack(
        (filtered_v[edge_index[0]], filtered_v[edge_index[1]])),
        axis=0)
    impl1 = persistence_routine(filtered_v, edge_index)
    impl2 = persistent_homology_cpu.compute_persistence_homology(
        filtered_v, filtered_e, edge_index)[0]

    #  print(impl1)
    #  print(impl2)
    assert (impl1 == impl2).all()


def test_persistence_computation_batched():
    edge_index = torch.Tensor([[0, 0, 2, 3, 4], [2, 3, 3, 4, 1]]).long()
    edge_index = torch.cat([edge_index, edge_index + 5], 1)
    filtered_v = torch.Tensor([1., 1., 2., 3., 4.]).repeat(2).unsqueeze(1)
    filtered_e = torch.max(torch.stack(
        (filtered_v[edge_index[0]], filtered_v[edge_index[1]])),
        axis=0)[0].repeat(2, 1)
    vertex_slices = torch.Tensor([0, 5, 10]).long()
    edge_slices = torch.Tensor([0, 5, 10]).long()
    # impl1 = persistence_routine(filtered_v, edge_index)
    impl2 = persistent_homology_cpu.compute_persistence_homology_batched(
        filtered_v, filtered_e, edge_index, vertex_slices, edge_slices)


def test_persistence_computation_batched_mt():
    edge_index = torch.Tensor([[0, 0, 2, 3, 4], [2, 3, 3, 4, 1]]).long()
    edge_index = torch.cat([edge_index, edge_index + 5], 1)
    filtered_v = torch.Tensor(
        [[1., 1., 2., 3., 4.], [1., 1., 2., 3., 4.]]).transpose(0, 1).repeat(2, 1)
    filtered_e = torch.max(torch.stack(
        (filtered_v[edge_index[0]], filtered_v[edge_index[1]])),
        axis=0)[0]
    vertex_slices = torch.Tensor([0, 5, 10]).long()
    edge_slices = torch.Tensor([0, 5, 10]).long()
    impl_st = persistent_homology_cpu.compute_persistence_homology_batched(
        filtered_v, filtered_e, edge_index, vertex_slices, edge_slices)

    filtered_v = filtered_v.transpose(1, 0)
    filtered_e = filtered_e.transpose(1, 0)
    edge_index = edge_index.transpose(1, 0)
    impl_mt = persistent_homology_cpu.compute_persistence_homology_batched_mt(
        filtered_v, filtered_e, edge_index, vertex_slices, edge_slices)
    # Check if implementations are the same
    assert (impl_st[0] == impl_mt[0]).all()
    assert ((impl_st[1] == impl_mt[1]) | torch.isnan(impl_mt[1])).all()
    # Check if instances are the same
    assert (
        impl_mt[0][:, vertex_slices[0]:vertex_slices[1]] ==
        impl_mt[0][:, vertex_slices[1]:vertex_slices[2]]).all()
    # Check if filtrations are the same
    assert (impl_mt[0][0] == impl_mt[0][1]).all()
    assert (
        impl_mt[1][:, edge_slices[0]:edge_slices[1]] ==
        impl_mt[1][:, edge_slices[1]:edge_slices[2]]).all()
    # Check if filtrations are the same
    assert (impl_mt[1][0] == impl_mt[1][1]).all()
