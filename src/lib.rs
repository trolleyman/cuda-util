#![cfg_attr(feature="unstable", feature(specialization))]

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
