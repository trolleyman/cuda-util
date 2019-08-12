
use ndarray::Dimension;
use crate::*;


pub trait TensorTrait {
	type Elem: TensorElem;
	type Dim: Dimension;
	fn cpu(&self) -> CpuTensor<Self::Elem, Self::Dim>;
	fn gpu(&self) -> GpuTensor<Self::Elem, Self::Dim>;
}

#[derive(Debug)]
pub enum Tensor<T: TensorElem, D: Dimension> {
	CpuTensor(CpuTensor<T, D>),
	GpuTensor(GpuTensor<T, D>),
}
impl<T: TensorElem, D: Dimension> TensorTrait for Tensor<T, D> {
	type Elem = T;
	type Dim = D;

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
