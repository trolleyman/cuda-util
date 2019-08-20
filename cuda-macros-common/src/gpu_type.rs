
use self::GpuTypeEnum::*;

	
static CUDA_NUMBER_TYPES: &'static [GpuTypeEnum] = &[
	F32(0.0),
	F64(0.0),
	U8 (0),
	I8 (0),
	U16(0),
	I16(0),
	U32(0),
	I32(0),
	U64(0),
	I64(0),
];


/// Enumeration of possible number types that can be sent to CUDA.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum GpuTypeEnum {
	/// f32
	F32(f32),
	/// f64
	F64(f64),
	/// u8
	U8(u8),
	/// i8
	I8(i8),
	/// u16
	U16(u16),
	/// i16
	I16(i16),
	/// i32
	U32(u32),
	/// u32
	I32(i32),
	/// i64
	U64(u64),
	/// u64
	I64(i64),
}
impl GpuTypeEnum {
	/// Returns a static slice of all CUDA number types.
	pub fn types() -> &'static [GpuTypeEnum] {
		CUDA_NUMBER_TYPES
	}
	
	/// Returns the enum name of the type.
	pub fn enum_name(&self) -> &'static str {
		match self {
			F32(_) => "F32",
			F64(_) => "F64",
			U8 (_) => "U8",
			I8 (_) => "I8",
			U16(_) => "U16",
			I16(_) => "I16",
			U32(_) => "U32",
			I32(_) => "I32",
			U64(_) => "U64",
			I64(_) => "I64",
		}
	}
	
	/// Returns the Rust name of the type.
	pub fn rust_name(&self) -> &'static str {
		match self {
			F32(_) => "f32",
			F64(_) => "f64",
			U8 (_) => "u8",
			I8 (_) => "i8",
			U16(_) => "u16",
			I16(_) => "i16",
			U32(_) => "u32",
			I32(_) => "i32",
			U64(_) => "u64",
			I64(_) => "i64",
		}
	}
	
	/// Returns the C name of the type.
	pub fn c_name(&self) -> &'static str {
		match self {
			F32(_) => "float",
			F64(_) => "double",
			U8 (_) => "uint8_t",
			I8 (_) => "int8_t",
			U16(_) => "uint16_t",
			I16(_) => "int16_t",
			U32(_) => "uint32_t",
			I32(_) => "int32_t",
			U64(_) => "uint64_t",
			I64(_) => "int64_t",
		}
	}
}

/// Any number that can be sent to CUDA implements this.
pub trait GpuType: num::Num + Copy {
	fn gpu_type(&self) -> GpuTypeEnum;
}

macro_rules! impl_gpu_type {
	($ty:ty => $p:path) => {
		impl GpuType for $ty {
			fn gpu_type(&self) -> GpuTypeEnum {
				$p(*self)
			}
		}
	};
}

impl_gpu_type!(f32 => F32);
impl_gpu_type!(f64 => F64);
impl_gpu_type!(u8  => U8);
impl_gpu_type!(i8  => I8);
impl_gpu_type!(u16 => U16);
impl_gpu_type!(i16 => I16);
impl_gpu_type!(u32 => U32);
impl_gpu_type!(i32 => I32);
impl_gpu_type!(u64 => U64);
impl_gpu_type!(i64 => I64);

#[test]
fn test_name() {
	use crate::*;

	assert_eq!(f32::gpu_type().rust_name(), "f32");
	assert_eq!(f64::gpu_type().rust_name(), "f64");

	assert_eq!(u8 ::gpu_type().rust_name(), "u8");
	assert_eq!(i8 ::gpu_type().rust_name(), "i8");
	assert_eq!(u16::gpu_type().rust_name(), "u16");
	assert_eq!(i16::gpu_type().rust_name(), "i16");
	assert_eq!(u32::gpu_type().rust_name(), "u32");
	assert_eq!(i32::gpu_type().rust_name(), "i32");
	assert_eq!(u64::gpu_type().rust_name(), "u64");
	assert_eq!(i64::gpu_type().rust_name(), "i64");
}
