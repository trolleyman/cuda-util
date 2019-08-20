
use ndarray::prelude::*;
use crate::*;

pub(crate) mod func;
mod vec;

pub use vec::*;


// pub trait GpuStorage {
// 	type Elem;
// 	private_decl!();
// }

/// `n`-dimensional vector (stored on the GPU) that can be easily moved between host and device
#[derive(Debug, Clone)]
pub struct GpuTensor<T: GpuType + 'static, D: Dimension> {
	data: GpuVec<T>,
	dim: D,
	strides: D,
}
impl<T: GpuType, D: Dimension> TensorTrait for GpuTensor<T, D> {
	type Elem = T;
	type Dim = D;

	fn from_ndarray<S>(_array: ndarray::ArrayBase::<S, D>) -> Self where S: ndarray::Data<Elem=T> {
		unimplemented!()
	}

	fn cpu(&self) -> CpuTensor<T, D> {
		unimplemented!()
	}
	fn gpu(&self) -> GpuTensor<T, D> {
		self.clone()
	}
}
