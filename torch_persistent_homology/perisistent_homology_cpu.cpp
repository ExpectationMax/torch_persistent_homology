#include "ATen/core/function_schema.h"
#include "unionfind.hh"
#include <ATen/Parallel.h>
#include <iostream>
#include <torch/extension.h>

using namespace torch::indexing;

template <typename dtype> void set_to_one(at::TensorAccessor<dtype, 1> z) {
  at::parallel_for(0, z.size(0), 0, [&](int64_t begin, int64_t end) {
    for (auto i = begin; i < end; i++) {
      z[i] = 1.;
    }
  });
}

void set_to_one_tensor(torch::Tensor z) {
  AT_DISPATCH_FLOATING_TYPES(z.scalar_type(), "set_to_one", ([&] {
                               set_to_one<scalar_t>(z.accessor<scalar_t, 1>());
                             }));
}

torch::Tensor ones_tensor() {
  auto z = torch::zeros(100);
  AT_DISPATCH_FLOATING_TYPES(z.scalar_type(), "set_to_one", ([&] {
                               set_to_one<scalar_t>(z.accessor<scalar_t, 1>());
                             }));
  return z;
}

torch::Tensor uf_find(torch::Tensor parents, int u) {
  // Creating a single element tensor seems a bit hacky, but I didn't find an
  // alternative way to return a single element while supporting multiple
  // integer types.
  auto out = torch::empty(1, parents.options());
  AT_DISPATCH_INTEGRAL_TYPES(parents.scalar_type(), "uf_find", ([&] {
                               out[0] = UnionFind<scalar_t>::find(
                                   parents.accessor<scalar_t, 1>(),
                                   static_cast<scalar_t>(u));
                             }));
  return out[0];
}

void uf_merge(torch::Tensor parents, int u, int v) {
  AT_DISPATCH_INTEGRAL_TYPES(parents.scalar_type(), "uf_merge", ([&] {
                               UnionFind<scalar_t>::merge(
                                   parents.accessor<scalar_t, 1>(),
                                   static_cast<scalar_t>(u),
                                   static_cast<scalar_t>(v));
                             }));
}

std::tuple<torch::Tensor, torch::Tensor>
compute_persistence_homology(torch::Tensor filtered_v, torch::Tensor filtered_e,
                             torch::Tensor edge_index) {
  auto n_vertices = filtered_v.size(0);
  auto n_edges = filtered_e.size(0);
  auto parents = torch::arange(0, n_vertices);
  auto parents_data = parents.accessor<int64_t, 1>();
  auto persistence = torch::zeros({n_vertices, 2}, filtered_v.options());
  auto persistence1 = torch::zeros({n_edges, 2}, filtered_e.options());

  // Looks like the more eligant alternative is C++17 thus we will have to live
  // with this.
  auto sorted_out = filtered_e.sort();
  auto &&filtered_e_sorted = std::get<0>(sorted_out);
  auto &&sorted_indices = std::get<1>(sorted_out);

  persistence.index_put_({Ellipsis, 0}, filtered_v);
  // std::cout << "persistence:" << persistence << std::endl;
  auto unpaired_value = filtered_e_sorted.index({-1});

  for (auto i = 0; i < n_edges; i++) {
    auto cur_edge_index = sorted_indices[i].item<int64_t>();
    auto cur_edge_weight = filtered_e_sorted[i].item<float>();
    auto nodes = edge_index.index({Ellipsis, cur_edge_index});
    // std::cout << "nodes:" << nodes << std::endl;

    auto node1 = nodes[0].item<int64_t>();
    auto node2 = nodes[1].item<int64_t>();

    auto younger = UnionFind<int64_t>::find(parents_data, node1);
    auto older = UnionFind<int64_t>::find(parents_data, node2);
    if (younger == older) {
      persistence1.index_put_({cur_edge_index, 0}, cur_edge_weight);
      persistence1.index_put_({cur_edge_index, 1}, unpaired_value);
      continue;
    } else {
      if (filtered_v[younger].item<float>() < filtered_v[older].item<float>()) {
        // Flip older and younger, node1 and node 2
        auto tmp = younger;
        younger = older;
        older = tmp;
        tmp = node1;
        node1 = node2;
        node2 = tmp;
      }
    }

    persistence.index_put_({younger, 1}, cur_edge_weight);
    UnionFind<int64_t>::merge(parents_data, node1, node2);
  }
  // Collect roots
  auto is_root = parents == torch::arange(0, n_vertices, parents.options());
  auto root_values = filtered_v.index({is_root});
  persistence.index_put_({is_root, 0}, root_values);
  persistence.index_put_({is_root, 1}, unpaired_value);
  // persistence.index_put_({is_root, 1}, -1);
  return std::make_tuple(std::move(persistence), std::move(persistence1));
}

std::tuple<torch::Tensor, torch::Tensor> compute_persistence_homology_batched(
    torch::Tensor filtered_v, torch::Tensor filtered_e,
    torch::Tensor edge_index, torch::Tensor vertex_slices,
    torch::Tensor edge_slices) {

  auto batch_size = vertex_slices.size(0) - 1;
  auto n_nodes = filtered_v.size(0);
  auto n_edges = filtered_e.size(0);
  auto n_filtrations = filtered_v.size(1);

  auto parents = torch::arange(0, n_nodes, edge_index.options())
                     .unsqueeze(0)
                     .repeat({n_filtrations, 1});
  auto persistence =
      torch::zeros({n_filtrations, n_nodes, 2}, filtered_v.options());
  auto persistence1 =
      torch::zeros({n_filtrations, n_edges, 2}, filtered_v.options());
  auto vertex_slices_data = vertex_slices.accessor<int64_t, 1>();
  auto edge_slices_data = edge_slices.accessor<int64_t, 1>();
  for (auto i = 0; i < batch_size * n_filtrations; i++) {
    auto instance = i / n_filtrations;
    auto filtration = i % n_filtrations;
    auto vertex_slice =
        Slice(vertex_slices_data[instance], vertex_slices_data[instance + 1]);
    auto edge_slice =
        Slice(edge_slices_data[instance], edge_slices_data[instance + 1]);
    auto cur_vertices = filtered_v.index({vertex_slice, filtration});
    auto cur_edges = filtered_e.index({edge_slice, filtration});
    auto vertex_offset = vertex_slices_data[instance];
    auto cur_edge_indices =
        edge_index.index({Ellipsis, edge_slice}) - vertex_offset;
    auto cur_res =
        compute_persistence_homology(cur_vertices, cur_edges, cur_edge_indices);
    persistence.index_put_({filtration, vertex_slice}, std::get<0>(cur_res));
    persistence1.index_put_({filtration, edge_slice}, std::get<1>(cur_res));
  }
  // Below code does not work due to usage of at:Tensors in threads. Need to
  // rewrite inner part of loop to not use at:Tensor but only raw memory access.
  // at::parallel_for(
  //     0, batch_size * n_filtrations, 0, [&](int64_t begin, int64_t end) {
  //       for (auto i = begin; i < end; i++) {
  //         auto instance = i / n_filtrations;
  //         auto filtration = i % n_filtrations;
  //         auto cur_vertices = filtered_v.index({Slice(
  //             vertex_slices_data[instance], vertex_slices_data[instance +
  //             1])});
  //         auto vertex_offset = vertex_slices_data[instance];
  //         auto cur_edges = filtered_e.index({Slice(
  //             edge_slices_data[instance], edge_slices_data[instance +
  //             1])});
  //         auto cur_edge_indices =
  //             edge_index.index(
  //                 {Ellipsis, Slice(edge_slices_data[instance],
  //                                  edge_slices_data[instance + 1])}) -
  //             vertex_offset;
  //         persistence.index_put_(
  //             {filtration}, compute_persistence_homology(
  //                               cur_vertices, cur_edges,
  //                               cur_edge_indices));
  //       }
  //     });
  return std::make_tuple(persistence, persistence1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("set_to_one", &set_to_one_tensor, "Test inplace parallel set to one");
  m.def("ones_tensor", &ones_tensor, "Test parallel set to one");
  m.def("uf_find", &uf_find, "UnionFind find operation");
  m.def("uf_merge", &uf_merge, "UnionFind merge operation");
  m.def("compute_persistence_homology", &compute_persistence_homology,
        "Persistence routine");
  m.def("compute_persistence_homology_batched",
        &compute_persistence_homology_batched, "Persistence routine");
}
