
use std::mem;

use cuda::runtime as rt;

pub use cuda::runtime::{CudaMemcpyKind, CudaResult, CudaError};


/// Allocates buffer to hold `len` elements of `T`.
/// 
/// # Safety
/// Remember to handle `Drop` correctly.
/// 
/// # Panics
/// If overflow occurs
pub unsafe fn cuda_alloc_device<T>(len: usize) -> CudaResult<*mut T> {
	rt::cuda_alloc_device(len.checked_mul(mem::size_of::<T>()).expect("overflow")).map(|p| p as *mut T)
}

/// Frees device pointer
pub unsafe fn cuda_free_device<T>(ptr: &mut *mut T) -> CudaResult<()> {
	let ptr_copy = *ptr;
	*ptr = std::ptr::null_mut();
	if ptr_copy != std::ptr::null_mut() {
		rt::cuda_free_device(ptr_copy as *mut u8)
	} else {
		Ok(())
	}
}

/// Copies `len` elements of `T` from `src` buffer to `dst` buffer.
/// 
/// # Safety
/// Remember to handle `Drop` correctly.
/// 
/// # Panics
/// If overflow occurs
pub unsafe fn cuda_memcpy<T>(dst: *mut T, src: *const T, len: usize, kind: CudaMemcpyKind) -> CudaResult<()> {
	rt::cuda_memcpy(dst as *mut u8, src as *const u8, len.checked_mul(mem::size_of::<T>()).expect("overflow"), kind)
}

/// Reallocates buffer `ptr` to hold `len` elements of `T` and copies `old_len` elements from old to new buffer.
/// 
/// # Safety
/// Remember to handle `Drop` correctly.
/// 
/// # Panics
/// If overflow occurs
pub unsafe fn cuda_realloc<T>(ptr: &mut *mut T, old_len: usize, len: usize) -> CudaResult<()> {
	// Allocate new buffer
	let mut new_ptr = cuda_alloc_device::<T>(len)?;
	if old_len > 0 {
		// Copy old elements over
		cuda_memcpy(new_ptr, *ptr, old_len, CudaMemcpyKind::DeviceToDevice)?;
	}
	// Swap buffers
	mem::swap(ptr, &mut new_ptr);
	// Free old buffer
	cuda_free_device(&mut new_ptr)
}

/// Copies a value from the device to the host, and returns it.
/// 
/// # Safety
/// Remember to handle `Drop` correctly.
pub unsafe fn cuda_copy_value_from_device<T>(ptr: *const T) -> CudaResult<T> {
	let mut x = std::mem::MaybeUninit::<T>::uninit();
	cuda_memcpy(x.as_mut_ptr(), ptr, 1, CudaMemcpyKind::DeviceToHost)?;
	Ok(x.assume_init())
}

/// Copies a value from the host to the device.
/// 
/// # Safety
/// Remember to handle `Drop` correctly.
pub unsafe fn cuda_copy_value_to_device<T>(ptr: *mut T, value: &T) -> CudaResult<()> {
	cuda_memcpy(ptr, value as *const T, 1, CudaMemcpyKind::HostToDevice)
}
