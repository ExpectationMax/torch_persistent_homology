from setuptools import setup, Extension
from torch.utils import cpp_extension

torch_library_paths = cpp_extension.library_paths(cuda=False)


def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    setup_kwargs.update({
        'ext_modules': [
            cpp_extension.CppExtension(
                'torch_persistent_homology.persistent_homology_cpu',
                ['torch_persistent_homology/perisistent_homology_cpu.cpp'],
                extra_link_args=[
                    '-Wl,-rpath,' + library_path
                    for library_path in torch_library_paths]
            )
        ],
        'cmdclass': {
            'build_ext': cpp_extension.BuildExtension
        }
    })
