
use nd::Dimension;
use crate::*;


/// Set of operations that all `Tensor`s implement
pub trait TensorTrait: Sized + std::fmt::Debug + Clone {
	/// Element type
	type Elem: GpuType;
	/// Dimension type
	type Dim: Dimension;

	/// Constructs the `Tensor` from an `ndarray` array
	fn from_ndarray<S>(array: nd::ArrayBase::<S, Self::Dim>) -> Self where S: nd::Data<Elem=Self::Elem>;
	
	/// Constructs a generic `Tensor` from `self`
	fn into_generic_tensor(self) -> Tensor<Self::Elem, Self::Dim>;

	/// Copy the `Tensor` to the CPU (host)
	fn cpu(&self) -> CpuTensor<Self::Elem, Self::Dim>;
	/// Copy the `Tensor` to the GPU (device)
	fn gpu(&self) -> GpuTensor<Self::Elem, Self::Dim>;

	/// Returns `true` if the tensor is stored on the CPU
	fn is_cpu(&self) -> bool;
	/// Returns `true` if the tensor is stored on the GPU
	fn is_gpu(&self) -> bool;
}

/// `n`-dimensional vector that can be easily moved between host and device
#[derive(Debug, Clone)]
pub enum Tensor<T: GpuType, D: Dimension> {
	CpuTensor(CpuTensor<T, D>),
	GpuTensor(GpuTensor<T, D>),
}
impl<T: GpuType, D: Dimension> TensorTrait for Tensor<T, D> {
	type Elem = T;
	type Dim = D;

	fn from_ndarray<S>(array: nd::ArrayBase::<S, D>) -> Self where S: nd::Data<Elem=T> {
		Tensor::CpuTensor(CpuTensor::from_ndarray(array))
	}

	fn into_generic_tensor(self) -> Tensor<T, D> {
		self
	}

	fn cpu(&self) -> CpuTensor<T, D> {
		match self {
			Tensor::CpuTensor(t) => t.cpu(),
			Tensor::GpuTensor(t) => t.cpu(),
		}
	}
	fn gpu(&self) -> GpuTensor<T, D> {
		match self {
			Tensor::CpuTensor(t) => t.gpu(),
			Tensor::GpuTensor(t) => t.gpu(),
		}
	}

	fn is_cpu(&self) -> bool {
		match self {
			Tensor::CpuTensor(_) => true,
			Tensor::GpuTensor(_) => false,
		}
	}
	fn is_gpu(&self) -> bool {
		match self {
			Tensor::CpuTensor(_) => false,
			Tensor::GpuTensor(_) => true,
		}
	}
}
