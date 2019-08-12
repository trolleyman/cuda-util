
use ndarray::prelude::*;
use crate::*;

mod vec;

pub use vec::*;


// pub trait GpuStorage {
// 	type Elem;
// 	private_decl!();
// }

#[derive(Debug, Clone)]
pub struct GpuTensor<T: TensorElem, D: Dimension> {
	data: GpuVec<T>,
	dim: D,
	strides: D,
}
impl<T: TensorElem, D: Dimension> TensorTrait for GpuTensor<T, D> {
	type Elem = T;
	type Dim = D;

	fn cpu(&self) -> CpuTensor<T, D> {
		unimplemented!()
	}
	fn gpu(&self) -> GpuTensor<T, D> {
		self.clone()
	}
}
