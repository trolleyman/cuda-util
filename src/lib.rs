#![cfg_attr(feature="unstable", feature(specialization))]
#![doc(html_logo_url = "https://raw.githubusercontent.com/trolleyman/cuda-util/master/res/logo_512.png")]

extern crate num;
extern crate ndarray;

extern crate cuda_macros;

#[macro_use]
mod private;
mod rcuda;

mod tensor;
pub use tensor::*;
mod cpu;
pub use cpu::*;
mod gpu;
pub use gpu::*;

pub use cuda_macros::*;


pub mod prelude {
	//! Useful items needed for all crates using `cuda_util`
	#[doc(no_inline)]
	pub use cuda_macros::ExecutionConfig;
	#[doc(no_inline)]
	pub use cuda_macros::{Dim3, dim3};
	
	#[doc(no_inline)]
	pub use super::{GpuVec, GpuSlice};
	#[doc(no_inline)]
	pub use super::{Tensor, TensorTrait, CpuTensor, GpuTensor};
}
