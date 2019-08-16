
use ndarray::Dimension;
use crate::*;


/// Set of operations that all `Tensor`s implement
pub trait TensorTrait: Sized {
	/// Element type
	type Elem: CudaNumber;
	/// Dimension type
	type Dim: Dimension;

	/// Constructs the `Tensor` from an `ndarray` array
	fn from_ndarray<S>(array: ndarray::ArrayBase::<S, Self::Dim>) -> Self where S: ndarray::Data<Elem=Self::Elem>;

	/// Copy the `Tensor` to the CPU (host)
	fn cpu(&self) -> CpuTensor<Self::Elem, Self::Dim>;
	/// Copy the `Tensor` to the GPU (device)
	fn gpu(&self) -> GpuTensor<Self::Elem, Self::Dim>;
}

/// `n`-dimensional vector that can be easily moved between host and device
#[derive(Debug)]
pub enum Tensor<T: CudaNumber + 'static, D: Dimension> {
	CpuTensor(CpuTensor<T, D>),
	GpuTensor(GpuTensor<T, D>),
}
impl<T: CudaNumber, D: Dimension> TensorTrait for Tensor<T, D> {
	type Elem = T;
	type Dim = D;

	fn from_ndarray<S>(array: ndarray::ArrayBase::<S, D>) -> Self where S: ndarray::Data<Elem=T> {
		Tensor::CpuTensor(CpuTensor::from_ndarray(array))
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
}
