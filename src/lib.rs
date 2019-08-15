#![cfg_attr(feature="specialization", feature(specialization))]

extern crate num;
extern crate ndarray;

extern crate cuda_macros;

mod rcuda;

mod tensor;
mod cpu;
mod gpu;

pub use cuda_macros::*;
pub use tensor::*;
pub use cpu::*;
pub use gpu::*;
