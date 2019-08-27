
use nd::prelude::*;
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
pub struct GpuTensor<T: GpuType, D: Dimension> {
	data: GpuVec<T>,
	dim: D,
	strides: D,
}
impl<T: GpuType, D: Dimension> TensorTrait for GpuTensor<T, D> {
	type Elem = T;
	type Dim = D;

	fn from_ndarray<S>(array: nd::ArrayBase::<S, D>) -> Self where S: nd::Data<Elem=T> {
		let (data, dim, strides) = func::gpu_vec_from_ndarray(array);
		GpuTensor {
			data,
			dim,
			strides,
		}
	}

	fn into_generic_tensor(self) -> Tensor<T, D> {
		Tensor::GpuTensor(self)
	}

	fn cpu(&self) -> CpuTensor<T, D> {
		CpuTensor::from_ndarray(Array::from_vec(self.data.to_vec()).into_shape(self.dim.clone()).unwrap())
	}
	fn gpu(&self) -> GpuTensor<T, D> {
		self.clone()
	}

	fn is_cpu(&self) -> bool {
		false
	}
	fn is_gpu(&self) -> bool {
		true
	}
}
