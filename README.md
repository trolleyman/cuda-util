
# `cuda-util`
Collection of utility methods & structs for working with the CUDA language.

**Main types:**
- `GpuVec` is the equivalent of a `Vec`, but is stored on the GPU.
- `Tensor` is an n-dimensional array type that can be easily transferred to and from the GPU.

## `cuda-macros`
The main purpose of `cuda-macros` is to define three attributes that perform the same
function as the `__global__`, `__device__` and `__host__` attributes do in the C
language.

To use these macros, a `build.rs` file must be defined that runs the `build()` function
in the `cuda-macros-build` crate. See the `cuda-macros-test` crate for an example crate.

Attribute(s) | CPU callable | GPU callable | Runs on
-------------|--------------|--------------|--------
`#[global]`  | ✔️ | ✔️️️️️️ | GPU |
`#[device]`  | ❌ | ️️️✔️️️️️️ | GPU |
`#[host]`    | ✔️ | ❌ | CPU |

The `#[host]` and `#[device]` attributes can be combined, in which case two versions will be generated, one for the host and one for the device.
The function will be run on whichever system it was called from.

In CUDA terminology, the CPU is the host, and the GPU is the device.
