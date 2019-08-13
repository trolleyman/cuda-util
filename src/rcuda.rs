
use std::mem;

use cuda::runtime as rt;

pub use cuda::runtime::{CudaMemcpyKind, CudaResult, CudaError};


/// Allocates buffer to hold `len` elements of `T`
/// 
/// # Panics
/// If overflow occurs
pub unsafe fn cuda_alloc_device<T: Copy>(len: usize) -> CudaResult<*mut T> {
	rt::cuda_alloc_device(len.checked_mul(mem::size_of::<T>()).expect("overflow")).map(|p| p as *mut T)
}

/// Frees device pointer
pub unsafe fn cuda_free_device<T: Copy>(ptr: *mut T) -> CudaResult<()> {
	if ptr != std::ptr::null_mut() {
		rt::cuda_free_device(ptr as *mut u8)
	} else {
		Ok(())
	}
}

/// Copies `len` elements of `T` from `src` buffer to `dst` buffer
/// 
/// # Panics
/// If overflow occurs
pub unsafe fn cuda_memcpy<T: Copy>(dst: *mut T, src: *const T, len: usize, kind: CudaMemcpyKind) -> CudaResult<()> {
	rt::cuda_memcpy(dst as *mut u8, src as *const u8, len.checked_mul(mem::size_of::<T>()).expect("overflow"), kind)
}

/// Reallocates buffer `ptr` to hold `len` elements of `T` and copies `old_len` elements from old to new buffer.
/// 
/// # Panics
/// If overflow occurs
pub unsafe fn cuda_realloc<T: Copy>(ptr: &mut *mut T, old_len: usize, len: usize) -> CudaResult<()> {
	// Allocate new buffer
	let mut new_ptr = cuda_alloc_device::<T>(len)?;
	if old_len > 0 {
		// Copy old elements over
		cuda_memcpy(new_ptr, *ptr, old_len, CudaMemcpyKind::DeviceToDevice)?;
	}
	// Swap buffers
	mem::swap(ptr, &mut new_ptr);
	// Free old buffer
	cuda_free_device(new_ptr)
}

/// Copies a value from the device to the host, and returns it.
pub unsafe fn cuda_copy_value_from_device<T: Copy>(ptr: *const T) -> CudaResult<T> {
	let mut x = std::mem::MaybeUninit::<T>::uninit();
	cuda_memcpy(x.as_mut_ptr(), ptr, 1, CudaMemcpyKind::DeviceToHost)?;
	Ok(x.assume_init())
}

/// Copies a value from the host to the device.
pub unsafe fn cuda_copy_value_to_device<T: Copy>(ptr: *mut T, value: T) -> CudaResult<()> {
	cuda_memcpy(ptr, &value as *const T, 1, CudaMemcpyKind::HostToDevice)
}
