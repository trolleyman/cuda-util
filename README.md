
# `cuda-util`
Collection of utility methods & structs for working with the CUDA language.

#### `#[global]`, `#[device]` and `#[host]`
These three attributes can be defined on a function to perform the same tasks
as the `__global__`, `__device__` and `__host__` attributes perform on C functions.

Attribute(s) | CPU callable | GPU callable | Runs on
-------------|--------------|--------------|--------
`#[global]`  | ✔️ | ✔️️️️️️ | GPU |
`#[device]`  | ❌ | ️️️✔️️️️️️ | GPU |
`#[host]`    | ✔️ | ❌ | CPU |

The `#[host]` and `#[device]` attributes can be combined, in which case two versions will be generated, one for the host and one for the device.
The function will be run on whichever system it was called from.

**TODO**: Mention `__CUDA_ARCH__` & implement `#[cfg(cuda_arch)]`

In CUDA terminology, the CPU is the host, and the GPU is the device.
