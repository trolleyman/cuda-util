
/// Enumeration of possible number types that can be sent to CUDA.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum CudaNumberType {
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
impl CudaNumberType {
	/// Returns the Rust name for the tensor element type.
	pub fn name(&self) -> &'static str {
		match self {
			CudaNumberType::F32 => "f32",
			CudaNumberType::F64 => "f64",
			CudaNumberType::U8  => "u8",
			CudaNumberType::I8  => "i8",
			CudaNumberType::U16 => "u16",
			CudaNumberType::I16 => "i16",
			CudaNumberType::U32 => "u32",
			CudaNumberType::I32 => "i32",
			CudaNumberType::U64 => "u64",
			CudaNumberType::I64 => "i64",
		}
	}
}

pub trait CudaNumber: num::Num + Copy {
	fn tensor_elem_type() -> CudaNumberType;
}

macro_rules! impl_cuda_number {
	($ty:ty => $ex:expr) => {
		impl CudaNumber for $ty {
			fn tensor_elem_type() -> CudaNumberType {
				$ex
			}
		}
	};
}

impl_cuda_number!(f32 => CudaNumberType::F32);
impl_cuda_number!(f64 => CudaNumberType::F64);

impl_cuda_number!(u8  => CudaNumberType::U8);
impl_cuda_number!(i8  => CudaNumberType::I8);
impl_cuda_number!(u16 => CudaNumberType::U16);
impl_cuda_number!(i16 => CudaNumberType::I16);
impl_cuda_number!(u32 => CudaNumberType::U32);
impl_cuda_number!(i32 => CudaNumberType::I32);
impl_cuda_number!(u64 => CudaNumberType::U64);
impl_cuda_number!(i64 => CudaNumberType::I64);

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
