#![cfg_attr(feature="unstable", feature(specialization))]
#![doc(html_logo_url = "https://raw.githubusercontent.com/trolleyman/cuda-util/master/res/logo_512.png")]

extern crate num;
extern crate ndarray;

extern crate cuda_macros;

mod rcuda;
#[macro_use]
mod private;
mod tensor;
mod cpu;
mod gpu;

pub use cuda_macros::*;
pub use tensor::*;
pub use cpu::*;
pub use gpu::*;

pub mod prelude {
	pub use super::{GpuVec, GpuSlice};
	pub use super::ExecutionConfig;
	pub use super::{Dim3, dim3};
	
	pub use super::{Tensor, TensorTrait, CpuTensor, GpuTensor};
}
