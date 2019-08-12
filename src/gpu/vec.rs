
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};


#[derive(Debug)]
pub struct GpuSlice<T> {
	_type: PhantomData<T>,
	ptr: *mut libc::c_void,
	len: usize,
}
impl<T> GpuSlice<T> {
	pub unsafe fn from_raw_parts<'a>(ptr: *const libc::c_void, len: usize) -> &'a Self {
		&GpuSlice {
			_type: PhantomData,
			ptr: ptr as *mut libc::c_void,
			len,
		}
	}

	pub unsafe fn from_raw_parts_mut<'a>(ptr: *mut libc::c_void, len: usize) -> &'a mut Self {
		&mut GpuSlice {
			_type: PhantomData,
			ptr,
			len,
		}
	}
}

#[derive(Debug)]
pub struct GpuVec<T> {
	_type: PhantomData<T>,
	ptr: *mut libc::c_void,
	len: usize,
	capacity: usize,
}
impl<T> Deref for GpuVec<T> {
	type Target = GpuSlice<T>;

	fn deref(&self) -> &GpuSlice<T> {
		GpuSlice::from_raw_parts(self.ptr, self.len)
	}
}
impl<T> DerefMut for GpuVec<T> {
	fn deref_mut(&mut self) -> &mut GpuSlice<T> {
		GpuSlice::from_raw_parts_mut(self.ptr, self.len)
	}
}
