
extern crate num;
extern crate ndarray;

extern crate cuda_macros;

mod rcuda;

mod elem;
mod tensor;
mod cpu;
mod gpu;

pub use cuda_macros::*;
pub use elem::*;
pub use tensor::*;
pub use cpu::*;
pub use gpu::*;
