#![doc(html_logo_url = "https://raw.githubusercontent.com/trolleyman/cuda-util/master/cuda-macros/res/logo_512.png")]

extern crate cuda_macros_common;
extern crate cuda_macros_impl;

pub use cuda_macros_impl::{host, device, global};
pub use cuda_macros_common::{dim3, GpuType, GpuTypeEnum, Dim3, ExecutionConfig};
