
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::mem::ManuallyDrop;
use std::fmt;
use std::convert::{AsRef, AsMut};

use crate::rcuda::*;


/// Reference to a device location
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
		writeln!(f, "GpuRef(<{:p}>)", self.ptr)
	}
}


/// Mutable reference to a device location
/// 
/// Values that are written to this are cached until this is dropped, or the [flush()](#method.flush) is called, at which point
/// the value is written to the device
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

	/// Copies the cached value to the pointer location given
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
		writeln!(f, "GpuMutRef(<{:p}>)", self.ptr)
	}
}

impl<'a, T> Drop for GpuMutRef<'a, T> {
	/// Flushes the value to the device.
	fn drop(&mut self) {
		if self.ptr != std::ptr::null_mut() {
			unsafe {
				cuda_copy_value_to_device(self.ptr, &*self.value).ok();
			}
		}
	}
}
