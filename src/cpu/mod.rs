
use ndarray::prelude::*;
use crate::*;


#[derive(Debug, Clone)]
pub struct CpuTensor<T: TensorElem, D: Dimension> {
	inner: Array<T, D>
}
impl<T: TensorElem, D: Dimension> TensorTrait for CpuTensor<T, D> {
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