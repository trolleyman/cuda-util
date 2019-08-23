//! Reference types that allow for easy modification of a `GpuVec` or `GpuSlice`.

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::mem::ManuallyDrop;
use std::fmt;
use std::convert::{AsRef, AsMut};

use crate::rcuda::*;


/// Reference to a slice of values held on the device, but cached on the host.
pub struct GpuSliceRef<'a, T> {
	_phantom: PhantomData<&'a T>,
	ptr: *const T,
	values: Vec<ManuallyDrop<T>>,
}
unsafe impl<'a, T: Send> Send for GpuSliceRef<'a, T> {}
unsafe impl<'a, T: Sync> Sync for GpuSliceRef<'a, T> {}
impl<'a, T> GpuSliceRef<'a, T> {
	/// Constructs a new `GpuSliceRef<'a, T>`.
	/// 
	/// # Safety
	/// `ptr` must be a valid device pointer that lasts for the lifetime `'a`.
	pub unsafe fn new(ptr: *const T, values: Vec<ManuallyDrop<T>>) -> Self {
		GpuSliceRef {
			_phantom: PhantomData,
			ptr,
			values
		}
	}

	/// Constructs a new `GpuSliceRef<'a, T>`.
	/// 
	/// # Safety
	/// `ptr` must be a valid device pointer that lasts for the lifetime `'a`.
	pub unsafe fn from_device_ptr(ptr: *const T, len: usize) -> CudaResult<Self> {
		let mut values: Vec<ManuallyDrop<T>> = Vec::with_capacity(len);
		cuda_memcpy(values.as_mut_ptr() as *mut T, ptr, len, CudaMemcpyKind::DeviceToHost)?;
		values.set_len(len);
		Ok(GpuSliceRef::new(ptr, values))
	}

	/// Return the device pointer
	pub fn device_ptr(&self) -> *const T {
		self.ptr
	}

	/// Gets a reference to the slice of values
	pub fn get(&self) -> &[T] {
		self.deref()
	}
}

impl<'a, T> Deref for GpuSliceRef<'a, T> {
	type Target = [T];
	fn deref(&self) -> &[T] {
		unsafe { std::mem::transmute(self.values.as_slice()) }
	}
}

impl<'a, T> AsRef<[T]> for GpuSliceRef<'a, T> {
	fn as_ref(&self) -> &[T] {
		self.deref()
	}
}

impl<'a, T> fmt::Debug for GpuSliceRef<'a, T> {
	fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
		write!(f, "GpuSliceRef([<{:p}>; {}])", self.ptr, self.len())
	}
}

impl<'a, T> fmt::Pointer for GpuSliceRef<'a, T> {
	fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
		fmt::Pointer::fmt(&self.ptr, f)
	}
}


/// Mutable reference to a slice of values held on the device, but cached on the host.
/// 
/// Values that are written to this are cached until this is dropped, or the [flush()](#method.flush) is called, at which point
/// the value is written to the device.
pub struct GpuSliceMutRef<'a, T> {
	_phantom: PhantomData<&'a T>,
	ptr: *mut T,
	values: Vec<ManuallyDrop<T>>,
}
unsafe impl<'a, T: Send> Send for GpuSliceMutRef<'a, T> {}
unsafe impl<'a, T: Sync> Sync for GpuSliceMutRef<'a, T> {}
impl<'a, T> GpuSliceMutRef<'a, T> {
	/// Constructs a new `GpuSliceMutRef<'a, T>`.
	/// 
	/// # Safety
	/// `ptr` must be a valid device pointer that lasts for the lifetime `'a`.
	pub unsafe fn new(ptr: *mut T, values: Vec<ManuallyDrop<T>>) -> Self {
		GpuSliceMutRef {
			_phantom: PhantomData,
			ptr,
			values
		}
	}

	/// Constructs a new `GpuSliceMutRef<'a, T>`.
	/// 
	/// # Safety
	/// `ptr` must be a valid mutable device pointer that lasts for the lifetime `'a`.
	/// 
	/// It must have **exclusive** access to it's memory for the entire lifetime.
	pub unsafe fn from_device_ptr(ptr: *mut T, len: usize) -> CudaResult<Self> {
		let mut values: Vec<ManuallyDrop<T>> = Vec::with_capacity(len);
		cuda_memcpy(values.as_mut_ptr() as *mut T, ptr, len, CudaMemcpyKind::DeviceToHost)?;
		values.set_len(len);
		Ok(GpuSliceMutRef::new(ptr, values))
	}

	/// Return the device pointer
	pub fn device_ptr(&self) -> *const T {
		self.ptr
	}

	/// Return the mutable device pointer
	pub fn device_ptr_mut(&mut self) -> *mut T {
		self.ptr
	}

	/// Gets a reference to the slice of values
	pub fn get(&self) -> &[T] {
		self.deref()
	}

	/// Gets a mutable reference to the slice of values
	pub fn get_mut(&mut self) -> &mut [T] {
		self.deref_mut()
	}

	/// Flushes the cached values back to the device
	pub fn flush(mut self) -> CudaResult<()> {
		let ptr = self.ptr;
		self.ptr = std::ptr::null_mut();
		if ptr != std::ptr::null_mut() {
			unsafe {
				cuda_memcpy(ptr, self.values.as_ptr() as *const T, self.values.len(), CudaMemcpyKind::HostToDevice)
			}
		} else {
			Ok(())
		}
	}
}

impl<'a, T> Deref for GpuSliceMutRef<'a, T> {
	type Target = [T];
	fn deref(&self) -> &[T] {
		unsafe { std::mem::transmute(self.values.as_slice()) }
	}
}

impl<'a, T> DerefMut for GpuSliceMutRef<'a, T> {
	fn deref_mut(&mut self) -> &mut [T] {
		unsafe { std::mem::transmute(self.values.as_mut_slice()) }
	}
}

impl<'a, T> AsRef<[T]> for GpuSliceMutRef<'a, T> {
	fn as_ref(&self) -> &[T] {
		self.deref()
	}
}

impl<'a, T> AsMut<[T]> for GpuSliceMutRef<'a, T> {
	fn as_mut(&mut self) -> &mut [T] {
		self.deref_mut()
	}
}

impl<'a, T> fmt::Debug for GpuSliceMutRef<'a, T> {
	fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
		write!(f, "GpuSliceMutRef([<{:p}>; {}])", self.ptr, self.len())
	}
}

impl<'a, T> fmt::Pointer for GpuSliceMutRef<'a, T> {
	fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
		fmt::Pointer::fmt(&self.ptr, f)
	}
}

impl<'a, T> Drop for GpuSliceMutRef<'a, T> {
	/// Flushes the value to the device, panicing on CUDA errors.
	fn drop(&mut self) {
		let ptr = self.ptr;
		self.ptr = std::ptr::null_mut();
		if ptr != std::ptr::null_mut() {
			unsafe {
				cuda_memcpy(ptr, self.values.as_ptr() as *const T, self.values.len(), CudaMemcpyKind::HostToDevice).expect("CUDA error");
			}
		}
	}
}


/// Reference to a value held on the device, but cached on the host.
pub struct GpuRef<'a, T> {
	_phantom: PhantomData<&'a T>,
	ptr: *const T,
	value: ManuallyDrop<T>
}
unsafe impl<'a, T: Send> Send for GpuRef<'a, T> {}
unsafe impl<'a, T: Sync> Sync for GpuRef<'a, T> {}
impl<'a, T> GpuRef<'a, T> {
	/// Constructs a new `GpuRef<'a, T>`.
	/// 
	/// # Safety
	/// `ptr` must be a valid device pointer that lasts for the lifetime `'a`.
	pub unsafe fn new(ptr: *const T, value: ManuallyDrop<T>) -> Self {
		GpuRef {
			_phantom: PhantomData,
			ptr,
			value
		}
	}

	/// Constructs a new `GpuRef<'a, T>`.
	/// 
	/// # Safety
	/// `ptr` must be a valid device pointer that lasts for the lifetime `'a`.
	pub unsafe fn from_device_ptr(ptr: *const T) -> CudaResult<Self> {
		Ok(GpuRef::new(ptr, ManuallyDrop::new(cuda_copy_value_from_device(ptr)?)))
	}

	/// Return the device pointer
	pub fn device_ptr(&self) -> *const T {
		self.ptr
	}

	/// Gets a reference to the value
	pub fn get(&self) -> &T {
		self.deref()
	}
}

impl<'a, T> Deref for GpuRef<'a, T> {
	type Target = T;
	fn deref(&self) -> &T {
		&self.value
	}
}

impl<'a, T> AsRef<T> for GpuRef<'a, T> {
	fn as_ref(&self) -> &T {
		self.deref()
	}
}

impl<'a, T> fmt::Debug for GpuRef<'a, T> {
	fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
		write!(f, "GpuRef(<{:p}>)", self.ptr)
	}
}

impl<'a, T> fmt::Pointer for GpuRef<'a, T> {
	fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
		fmt::Pointer::fmt(&self.ptr, f)
	}
}


/// Mutable reference to a value held on the device, but cached on the host.
/// 
/// Values that are written to this are cached until this is dropped, or the [flush()](#method.flush) is called, at which point
/// the value is written to the device.
pub struct GpuMutRef<'a, T> {
	_phantom: PhantomData<&'a T>,
	ptr: *mut T,
	value: ManuallyDrop<T>,
}
unsafe impl<'a, T: Send> Send for GpuMutRef<'a, T> {}
unsafe impl<'a, T: Sync> Sync for GpuMutRef<'a, T> {}
impl<'a, T> GpuMutRef<'a, T> {
	/// Constructs a new `GpuMutRef<'a, T>`.
	/// 
	/// # Safety
	/// `ptr` must be a valid device pointer that lasts for the lifetime `'a`.
	pub unsafe fn new(ptr: *mut T, value: ManuallyDrop<T>) -> Self {
		GpuMutRef {
			_phantom: PhantomData,
			ptr,
			value
		}
	}

	/// Constructs a new `GpuMutRef<'a, T>`.
	/// 
	/// # Safety
	/// `ptr` must be a valid mutable device pointer that lasts for the lifetime `'a`.
	/// 
	/// It must have **exclusive** access to it's memory for the entire lifetime.
	pub unsafe fn from_device_ptr(ptr: *mut T) -> CudaResult<Self> {
		Ok(GpuMutRef::new(ptr, ManuallyDrop::new(cuda_copy_value_from_device(ptr)?)))
	}

	/// Return the device pointer
	pub fn device_ptr(&self) -> *const T {
		self.ptr
	}

	/// Return the mutable device pointer
	pub fn device_ptr_mut(&mut self) -> *mut T {
		self.ptr
	}

	/// Gets a reference to the value
	pub fn get(&self) -> &T {
		self.deref()
	}

	/// Gets a mutable reference to the value
	pub fn get_mut(&mut self) -> &mut T {
		self.deref_mut()
	}

	/// Writes the value given into the cache
	pub fn write(&mut self, value: T) {
		unsafe {
			ManuallyDrop::drop(&mut self.value);
		}
		*self.value = value;
	}

	/// Flushes the cached value back to the device
	pub fn flush(mut self) -> CudaResult<()> {
		let ptr = self.ptr;
		self.ptr = std::ptr::null_mut();
		if ptr != std::ptr::null_mut() {
			unsafe {
				cuda_copy_value_to_device(ptr, &*self.value)
			}
		} else {
			Ok(())
		}
	}
}

impl<'a, T> Deref for GpuMutRef<'a, T> {
	type Target = T;
	fn deref(&self) -> &T {
		&*self.value
	}
}

impl<'a, T> DerefMut for GpuMutRef<'a, T> {
	fn deref_mut(&mut self) -> &mut T {
		&mut *self.value
	}
}

impl<'a, T> AsRef<T> for GpuMutRef<'a, T> {
	fn as_ref(&self) -> &T {
		self.deref()
	}
}

impl<'a, T> AsMut<T> for GpuMutRef<'a, T> {
	fn as_mut(&mut self) -> &mut T {
		self.deref_mut()
	}
}

impl<'a, T> fmt::Debug for GpuMutRef<'a, T> {
	fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
		write!(f, "GpuMutRef(<{:p}>)", self.ptr)
	}
}

impl<'a, T> fmt::Pointer for GpuMutRef<'a, T> {
	fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
		fmt::Pointer::fmt(&self.ptr, f)
	}
}

impl<'a, T> Drop for GpuMutRef<'a, T> {
	/// Flushes the value to the device, panicing on CUDA errors.
	fn drop(&mut self) {
		if self.ptr != std::ptr::null_mut() {
			unsafe {
				cuda_copy_value_to_device(self.ptr, &*self.value).expect("CUDA error");
			}
		}
	}
}
