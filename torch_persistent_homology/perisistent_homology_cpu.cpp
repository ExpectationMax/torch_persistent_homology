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

torch::Tensor compute_persistence_homology(torch::Tensor filtered_v,
                                           torch::Tensor filtered_e,
                                           torch::Tensor edge_index) {
  auto n_vertices = filtered_v.size(0);
  auto n_edges = filtered_e.size(0);
  auto parents = torch::arange(0, n_vertices);
  auto parents_data = parents.accessor<int64_t, 1>();
  auto persistence =
      torch::zeros(at::IntArrayRef({n_vertices, 2}), filtered_v.options());
  auto persistence1 = torch::zeros({n_edges, 2}, filtered_e.options());

  // Looks like the more eligant alternative is C++17 thus we will have to live
  // with this.
  torch::Tensor filtered_e_sorted, sorted_indices;
  auto sorted_out = filtered_e.sort();
  filtered_e_sorted = std::get<0>(sorted_out);
  sorted_indices = std::get<1>(sorted_out);

  persistence.index_put_({Ellipsis, 0}, filtered_v);
  std::cout << "persistence:" << persistence << std::endl;
  auto unpaired_value = filtered_e.index({-1});

  for (auto i = 0; i < n_edges; i++) {
    auto cur_edge_index = sorted_indices[i].item<int64_t>();
    auto cur_edge_weight = filtered_e_sorted[i].item<float>();
    auto nodes = edge_index.index({Ellipsis, cur_edge_index});
    std::cout << "nodes:" << nodes << std::endl;

    auto node1 = nodes[0].item<int64_t>();
    auto node2 = nodes[1].item<int64_t>();

    auto younger = UnionFind<int64_t>::find(parents_data, node1);
    auto older = UnionFind<int64_t>::find(parents_data, node2);
    if (younger == older) {
      persistence1.index_put_({edge_index, 0}, cur_edge_weight);
      persistence1.index_put_({edge_index, 1}, unpaired_value);
      continue;
    } else {
      if (filtered_v[younger].item<float>() < filtered_v[older].item<float>()) {
        // Filp older and younger, node1 and node 2
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
  return persistence;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("set_to_one", &set_to_one_tensor, "Test inplace parallel set to one");
  m.def("ones_tensor", &ones_tensor, "Test parallel set to one");
  m.def("uf_find", &uf_find, "UnionFind find operation");
  m.def("uf_merge", &uf_merge, "UnionFind merge operation");
  m.def("persistence_routine", &compute_persistence_homology,
        "Persistence routine");
}
