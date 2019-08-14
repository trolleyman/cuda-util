
use ndarray::prelude::*;
use crate::*;


/// `n`-dimensional vector (stored on the CPU) that can be easily moved between host and device
#[derive(Debug, Clone)]
pub struct CpuTensor<T: CudaNumber, D: Dimension> {
	inner: Array<T, D>
}
impl<T: CudaNumber, D: Dimension> TensorTrait for CpuTensor<T, D> {
	type Elem = T;
	type Dim = D;
	
	fn from_ndarray<S>(array: ndarray::ArrayBase::<S, D>) -> Self where S: ndarray::Data<Elem=T> {
		CpuTensor {
			inner: array.to_owned()
		}
	}

	fn cpu(&self) -> CpuTensor<T, D> {
		self.clone()
	}
	fn gpu(&self) -> GpuTensor<T, D> {
		GpuTensor::from_ndarray(self.inner.view())
	}
}