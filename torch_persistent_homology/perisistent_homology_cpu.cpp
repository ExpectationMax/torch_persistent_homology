#include "unionfind.hh"
#include <ATen/Parallel.h>
#include <torch/extension.h>

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("set_to_one", &set_to_one_tensor, "Test inplace parallel set to one");
  m.def("ones_tensor", &ones_tensor, "Test parallel set to one");
  m.def("uf_find", &uf_find, "UnionFind find operation");
  m.def("uf_merge", &uf_merge, "UnionFind merge operation");
}
