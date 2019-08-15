
use self::CudaNumberType::*;
static CUDA_NUMBER_TYPES: &'static [CudaNumberType] = &[F32, F64, U8, I8, U16, I16, U32, I32, U64, I64];

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
	/// Returns a static slice of all CUDA number types.
	pub fn types() -> &'static [CudaNumberType] {
		CUDA_NUMBER_TYPES
	}
	
	/// Returns the Rust name for the tensor element type.
	pub fn rust_name(&self) -> &'static str {
		match self {
			F32 => "f32",
			F64 => "f64",
			U8  => "u8",
			I8  => "i8",
			U16 => "u16",
			I16 => "i16",
			U32 => "u32",
			I32 => "i32",
			U64 => "u64",
			I64 => "i64",
		}
	}
	
	/// Returns the C name for the tensor element type.
	pub fn c_name(&self) -> &'static str {
		match self {
			F32 => "float",
			F64 => "double",
			U8  => "uint8_t",
			I8  => "int8_t",
			U16 => "uint16_t",
			I16 => "int16_t",
			U32 => "uint32_t",
			I32 => "int32_t",
			U64 => "uint64_t",
			I64 => "int64_t",
		}
	}
}

/// Any number that can be sent to CUDA implements this.
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

impl_cuda_number!(f32 => F32);
impl_cuda_number!(f64 => F64);

impl_cuda_number!(u8  => U8);
impl_cuda_number!(i8  => I8);
impl_cuda_number!(u16 => U16);
impl_cuda_number!(i16 => I16);
impl_cuda_number!(u32 => U32);
impl_cuda_number!(i32 => I32);
impl_cuda_number!(u64 => U64);
impl_cuda_number!(i64 => I64);

#[test]
fn test_name() {
	use crate::*;

	assert_eq!(f32::tensor_elem_type().rust_name(), "f32");
	assert_eq!(f64::tensor_elem_type().rust_name(), "f64");

	assert_eq!(u8 ::tensor_elem_type().rust_name(), "u8");
	assert_eq!(i8 ::tensor_elem_type().rust_name(), "i8");
	assert_eq!(u16::tensor_elem_type().rust_name(), "u16");
	assert_eq!(i16::tensor_elem_type().rust_name(), "i16");
	assert_eq!(u32::tensor_elem_type().rust_name(), "u32");
	assert_eq!(i32::tensor_elem_type().rust_name(), "i32");
	assert_eq!(u64::tensor_elem_type().rust_name(), "u64");
	assert_eq!(i64::tensor_elem_type().rust_name(), "i64");
}
