
/// Enumeration of possible types that can be stored in a tensor.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum TensorElemType {
	/// f32
	F32,
	/// f64
	F64,
	/// u8
	U8,
	/// i8
	I8,
	/// u16
	U16,
	/// i16
	I16,
	/// i32
	U32,
	/// u32
	I32,
	/// i64
	U64,
	/// u64
	I64,
}
impl TensorElemType {
	/// Returns the Rust name for the tensor element type.
	pub fn name(&self) -> &'static str {
		match self {
			TensorElemType::F32 => "f32",
			TensorElemType::F64 => "f64",
			TensorElemType::U8  => "u8",
			TensorElemType::I8  => "i8",
			TensorElemType::U16 => "u16",
			TensorElemType::I16 => "i16",
			TensorElemType::U32 => "u32",
			TensorElemType::I32 => "i32",
			TensorElemType::U64 => "u64",
			TensorElemType::I64 => "i64",
		}
	}
}

pub trait TensorElem: num::Num + Copy {
	fn tensor_elem_type() -> TensorElemType;
}

macro_rules! impl_cuda_number {
	($ty:ty => $ex:expr) => {
		impl TensorElem for $ty {
			fn tensor_elem_type() -> TensorElemType {
				$ex
			}
		}
	};
}

impl_cuda_number!(f32 => TensorElemType::F32);
impl_cuda_number!(f64 => TensorElemType::F64);

impl_cuda_number!(u8  => TensorElemType::U8);
impl_cuda_number!(i8  => TensorElemType::I8);
impl_cuda_number!(u16 => TensorElemType::U16);
impl_cuda_number!(i16 => TensorElemType::I16);
impl_cuda_number!(u32 => TensorElemType::U32);
impl_cuda_number!(i32 => TensorElemType::I32);
impl_cuda_number!(u64 => TensorElemType::U64);
impl_cuda_number!(i64 => TensorElemType::I64);

#[test]
fn test_name() {
	use crate::*;

	assert_eq!(f32::tensor_elem_type().name(), "f32");
	assert_eq!(f64::tensor_elem_type().name(), "f64");

	assert_eq!(u8 ::tensor_elem_type().name(), "u8");
	assert_eq!(i8 ::tensor_elem_type().name(), "i8");
	assert_eq!(u16::tensor_elem_type().name(), "u16");
	assert_eq!(i16::tensor_elem_type().name(), "i16");
	assert_eq!(u32::tensor_elem_type().name(), "u32");
	assert_eq!(i32::tensor_elem_type().name(), "i32");
	assert_eq!(u64::tensor_elem_type().name(), "u64");
	assert_eq!(i64::tensor_elem_type().name(), "i64");
}
