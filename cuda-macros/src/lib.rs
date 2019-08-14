
extern crate cuda_macros_common;
extern crate cuda_macros_impl;

pub use cuda_macros_impl::{host, device, global};
pub use cuda_macros_common::{CudaNumber, ExecutionConfig};
