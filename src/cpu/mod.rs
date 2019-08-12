
use ndarray::prelude::*;
use crate::*;


#[derive(Debug, Clone)]
pub struct CpuTensor<T: TensorElem, D: Dimension> {
	inner: Array<T, D>
}
impl<T: TensorElem, D: Dimension> TensorTrait for CpuTensor<T, D> {
	type Elem = T;
	type Dim = D;

	fn cpu(&self) -> CpuTensor<T, D> {
		self.clone()
	}
	fn gpu(&self) -> GpuTensor<T, D> {
		GpuTensor::from_shape_slice(self.inner.raw_dim(), self.inner.iter())
	}
}