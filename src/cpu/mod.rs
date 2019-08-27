
use nd::prelude::*;
use crate::*;


/// `n`-dimensional vector (stored on the CPU) that can be easily moved between host and device
#[derive(Debug, Clone)]
pub struct CpuTensor<T: GpuType, D: Dimension> {
	inner: Array<T, D>
}
impl<T: GpuType, D: Dimension> TensorTrait for CpuTensor<T, D> {
	type Elem = T;
	type Dim = D;
	
	fn from_ndarray<S>(array: nd::ArrayBase::<S, D>) -> Self where S: nd::Data<Elem=T> {
		CpuTensor {
			inner: array.into_owned()
		}
	}

	fn into_generic_tensor(self) -> Tensor<T, D> {
		Tensor::CpuTensor(self)
	}

	fn cpu(&self) -> CpuTensor<T, D> {
		self.clone()
	}
	fn gpu(&self) -> GpuTensor<T, D> {
		GpuTensor::from_ndarray(self.inner.view())
	}

	fn is_cpu(&self) -> bool {
		true
	}
	fn is_gpu(&self) -> bool {
		false
	}
}
