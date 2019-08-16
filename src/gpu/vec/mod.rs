
use std::marker::PhantomData;
use std::ops::{self, Deref, DerefMut};
use std::mem::{ManuallyDrop, MaybeUninit, size_of};
use std::fmt;
use std::convert::{AsRef, AsMut};

use cfg_if::cfg_if;

mod index;
pub use self::index::*;
mod reference;
pub use self::reference::*;

use crate::rcuda::*;
use super::func;


/// Slice of [`GpuVec`](struct.GpuVec.html).
/// 
/// [`Unsize`](https://doc.rust-lang.org/std/marker/trait.Unsize.html)d type that can be only accessed via a reference (e.g. `&mut GpuSlice`).
/// 
/// Can be created by dereferencing a GpuVec, see [`GpuVec::deref()`](struct.GpuVec.html#method.deref).
pub struct GpuSlice<T>(ManuallyDrop<[T]>);
impl<T> GpuSlice<T> {
	/// Constructs an immutable GpuSlice from raw parts.
	/// 
	/// # Safety
	/// `data` must point to a device buffer of at least size `len` that lives for at least lifetime `'a`.
	pub unsafe fn from_raw_parts<'a>(data: *const T, len: usize) -> &'a Self {
		std::mem::transmute(std::slice::from_raw_parts(data, len))
	}

	/// Constructs a mutable GpuSlice from raw parts.
	/// 
	/// # Safety
	/// `data` must point to a device buffer of at least size `len` that lives for at least lifetime `'a`.
	pub unsafe fn from_raw_parts_mut<'a>(data: *mut T, len: usize) -> &'a mut Self {
		std::mem::transmute(std::slice::from_raw_parts_mut(data, len))
	}

	/// Copies the data in the slice to the CPU and returns it as a `Vec`.
	/// 
	/// This is a relatively slow operation, as it requires moving memory between the RAM and the GPU.
	/// 
	/// # Panics
	/// Panics if a CUDA error is encountered while performing this operation.
	pub fn copy_to_vec(&self) -> Vec<T> where T: Copy {
		let mut v = Vec::with_capacity(self.len());
		unsafe {
			cuda_memcpy(v.as_mut_ptr(), self.as_ptr(), self.len(), CudaMemcpyKind::DeviceToHost).expect("CUDA error");
			v.set_len(self.len());
		}
		v
	}

	/// Copies the data in the slice to the CPU and returns it as a `GpuSliceRef`.
	/// The lifetime of this reference is linked to the lifetime of the `GpuSlice`.
	/// 
	/// This is a relatively slow operation, as it requires moving memory between the RAM and the GPU.
	/// 
	/// # Panics
	/// Panics if a CUDA error is encountered while performing this operation.
	pub fn borrow_as_cpu_slice<'a>(&'a self) -> GpuSliceRef<'a, T> {
		GpuSliceRef::from_device_ptr(self.as_ptr(), self.len()).expect("CUDA error")
	}

	/// Copies the data in the slice to the CPU and returns it as a mutable `GpuSliceRef`.
	/// The lifetime of this reference is linked to the lifetime of the `GpuSlice`.
	/// 
	/// This is a relatively slow operation, as it requires moving memory between the RAM and the GPU.
	/// 
	/// # Panics
	/// Panics if a CUDA error is encountered while performing this operation.
	pub fn borrow_as_cpu_slice_mut<'a>(&'a mut self) -> GpuSliceMutRef<'a, T> {
		GpuSliceMutRef::from_device_ptr(self.as_mut_ptr(), self.len()).expect("CUDA error")
	}

	/// Clones the data in the slice to the CPU and returns it as a `Vec`.
	/// 
	/// This is a relatively slow operation, as it requires moving memory between the RAM and the GPU.
	/// 
	/// # Panics
	/// Panics if a CUDA error is encountered while performing this operation.
	pub fn clone_to_vec(&self) -> Vec<T> where T: Clone {
		let mut v: Vec<ManuallyDrop<T>> = Vec::with_capacity(self.len());
		let mut ret: Vec<T> = Vec::with_capacity(self.len());
		unsafe {
			// Copy to `v`
			cuda_memcpy(v.as_mut_ptr() as *mut T, self.as_ptr(), self.len(), CudaMemcpyKind::DeviceToHost).expect("CUDA error");
			v.set_len(self.len());
		}
		// Call `clone()` on each element in `v`
		for elem in &v {
			ret.push(elem.deref().clone());
		}
		ret
	}

	/// Copies the data in the slice to a new `GpuVec`.
	/// 
	/// # Panics
	/// Panics if a CUDA error is encountered while performing this operation.
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::*;
	/// let a = GpuVec::from(&[1, 2, 3, 4][..]);
	/// let b: GpuVec<_> = (&a[1..]).copy_to_gpu_vec();
	/// assert_eq!(b.into_vec(), &[2, 3, 4]);
	/// ```
	pub fn copy_to_gpu_vec(&self) -> GpuVec<T> where T: Copy {
		let mut v = GpuVec::with_capacity(self.len());
		unsafe {
			cuda_memcpy(v.as_mut_ptr(), self.as_ptr(), self.len(), CudaMemcpyKind::DeviceToDevice).expect("CUDA error");
			v.set_len(self.len());
		}
		v
	}

	/// Returns the number of elements in the slice
	pub fn len(&self) -> usize {
		self.0.len()
	}

	/// Returns `true` if the slice has a length of zero
	pub fn is_empty(&self) -> bool {
		self.0.is_empty()
	}

	/// Gets the first element in the slice
	/// 
	/// This is a relatively slow operation, as it requires moving memory between the RAM and the GPU.
	/// 
	/// # Panics
	/// Panics if a CUDA error is encountered while performing this operation.
	pub fn first<'a>(&'a self) -> Option<GpuRef<'a, T>> {
		if self.len() == 0 {
			None
		} else {
			unsafe {
				Some(GpuRef::from_device_ptr(self.as_ptr()).expect("CUDA error"))
			}
		}
	}

	/// Gets a mutable reference to the first element in the slice
	/// 
	/// This is a relatively slow operation, as it requires moving memory between the RAM and the GPU.
	/// 
	/// # Panics
	/// Panics if a CUDA error is encountered while performing this operation.
	pub fn first_mut<'a>(&'a mut self) -> Option<GpuMutRef<'a, T>> {
		if self.len() == 0 {
			None
		} else {
			unsafe {
				Some(GpuMutRef::from_device_ptr(self.as_mut_ptr()).expect("CUDA error"))
			}
		}
	}

	/// Gets the last element in the slice
	/// 
	/// This is a relatively slow operation, as it requires moving memory between the RAM and the GPU.
	/// 
	/// # Panics
	/// Panics if a CUDA error is encountered while performing this operation.
	pub fn last<'a>(&'a self) -> Option<GpuRef<'a, T>> {
		if self.len() == 0 {
			None
		} else {
			unsafe {
				Some(GpuRef::from_device_ptr(self.as_ptr().add(self.len() - 1)).expect("CUDA error"))
			}
		}
	}

	/// Gets a mutable reference to the last element in the slice
	/// 
	/// This is a relatively slow operation, as it requires moving memory between the RAM and the GPU.
	/// 
	/// # Panics
	/// Panics if a CUDA error is encountered while performing this operation.
	pub fn last_mut<'a>(&'a mut self) -> Option<GpuMutRef<'a, T>> {
		if self.len() == 0 {
			None
		} else {
			unsafe {
				Some(GpuMutRef::from_device_ptr(self.as_mut_ptr().add(self.len() - 1)).expect("CUDA error"))
			}
		}
	}

	/// Returns a reference to the element at position `index`, or `None` if out of bounds.
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::GpuVec;
	/// let mut v = GpuVec::from(&[1, 2, 100, 4, 5][..]);
	/// assert_eq!(v.get(2).unwrap(), 100);
	/// ```
	/// 
	/// # Panics
	/// Panics if a CUDA error is encountered while performing this operation. See [`try_get()`](#method.try_get).
	pub fn get<'a>(&'a self, index: usize) -> Option<GpuRef<'a, T>> {
		self.try_get(index).expect("CUDA error")
	}

	/// Returns a mutable reference to the element at position `index`, or `None` if out of bounds.
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::GpuVec;
	/// let mut v = GpuVec::from(&[1, 2, 3, 4, 5][..]);
	/// {
	/// 	let ref = v.get_mut(2).unwrap()
	/// 	*ref = 100;
	/// }
	/// assert_eq!(v.copy_to_vec(), &[1, 2, 100, 4, 5]);
	/// ```
	/// 
	/// # Panics
	/// Panics if a CUDA error is encountered while performing this operation. See [`try_get_mut()`](#method.try_get_mut).
	pub fn get_mut<'a>(&'a mut self, index: usize) -> Option<GpuMutRef<'a, T>> {
		self.try_get_mut(index).expect("CUDA error")
	}

	/// Returns a reference to the element at position `index`, `Ok(None)` if out of bounds,
	/// or `Err` if a CUDA error was encountered while performing this operation.
	pub fn try_get<'a>(&'a self, index: usize) -> CudaResult<Option<GpuRef<'a, T>>> {
		if index < self.len() {
			unsafe { Ok(Some(GpuRef::from_device_ptr(self.as_ptr().add(index))?)) }
		} else {
			Ok(None)
		}
	}

	/// Returns a mutable reference to the element at position `index`, `Ok(None)` if out of bounds,
	/// or `Err` if a CUDA error was encountered while performing this operation.
	pub fn try_get_mut<'a>(&'a mut self, index: usize) -> CudaResult<Option<GpuMutRef<'a, T>>> {
		if index < self.len() {
			unsafe { Ok(Some(GpuMutRef::from_device_ptr(self.as_mut_ptr().add(index))?)) }
		} else {
			Ok(None)
		}
	}

	/// Returns a raw pointer to the device buffer.
	/// 
	/// # Safety
	/// Note that this is *not* a valid pointer. This is a device pointer, that points to data on the GPU. It can only be used in CUDA functions that explicitly mention a requirement for a device pointer.
	/// 
	/// The caller is also responsible for the lifetime of the pointer.
	pub fn as_ptr(&self) -> *const T {
		self.0.as_ptr()
	}

	/// Returns a raw mutable pointer to the device buffer.
	/// 
	/// # Safety
	/// Note that this is *not* a valid pointer. This is a device pointer, that points to data on the GPU.
	/// It can only be used in CUDA functions that explicitly mention a requirement for a device pointer.
	/// 
	/// The caller is also responsible for the lifetime of the pointer.
	pub fn as_mut_ptr(&mut self) -> *mut T {
		self.0.as_mut_ptr()
	}

	/// Swaps two elements in the slice.
	/// 
	/// # Panics
	/// Panics if `a` or `b` are out of bounds,
	/// or if a CUDA error is encountered while performing this operation.
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::*;
	/// let mut v = GpuVec::from(&[1, 2, 3][..]);
	/// v.swap(0, 1);
	/// assert_eq!(v.copy_to_vec(), &[2, 1, 3]);
	/// ```
	pub fn swap(&mut self, a: usize, b: usize) {
		if a >= self.len() {
			panic!("a is out of bounds");
		}
		if b >= self.len() {
			panic!("b is out of bounds");
		}
		unsafe {
			let a_ptr = self.as_mut_ptr().add(a);
			let b_ptr = self.as_mut_ptr().add(b);
			let tmp = cuda_alloc_device::<T>(1).expect("CUDA error");
			cuda_memcpy(tmp, a_ptr, 1, CudaMemcpyKind::DeviceToDevice).expect("CUDA error");
			cuda_memcpy(a_ptr, b_ptr, 1, CudaMemcpyKind::DeviceToDevice).expect("CUDA error");
			cuda_memcpy(b_ptr, tmp, 1, CudaMemcpyKind::DeviceToDevice).expect("CUDA error");
			cuda_free_device(tmp).expect("CUDA error");
		}
	}

	/// Reverses the order of the elements in the slice.
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::*;
	/// let mut cpu_vec: Vec<u32> = (0..2000).into_iter().collect();
	/// let mut v = GpuVec::from(&cpu_vec[..]);
	/// v.reverse();
	/// cpu_vec.reverse();
	/// assert_eq!(&v.copy_to_vec(), &cpu_vec);
	/// ```
	pub fn reverse(&mut self) {
		func::reverse_vector(self.as_mut_ptr(), self.len());
	}

	// TODO: Everything in https://doc.rust-lang.org/std/primitive.slice.html below reverse()
}

impl<T> fmt::Debug for GpuSlice<T> {
	fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
		write!(f, "GpuSlice([<{:p}>; {}])", self.as_ptr(), self.len())
	}
}

impl<T, I: GpuSliceRange<T>> ops::Index<I> for GpuSlice<T> {
	type Output = GpuSlice<T>;
	fn index(&self, index: I) -> &GpuSlice<T> {
		index.slice(self)
	}
}
impl<T, I: GpuSliceRange<T>> ops::IndexMut<I> for GpuSlice<T> {
	fn index_mut(&mut self, index: I) -> &mut GpuSlice<T> {
		index.slice_mut(self)
	}
}

// TODO: Slice ops: https://doc.rust-lang.org/std/primitive.slice.html

/// Contiguous growable array type, similar to `Vec<T>` but stored on the GPU.
#[derive(Debug)]
pub struct GpuVec<T> {
	_type: PhantomData<T>,
	ptr: *mut T,
	len: usize,
	capacity: usize,
}
unsafe impl<T: Send> Send for GpuVec<T> {}
unsafe impl<T: Sync> Sync for GpuVec<T> {}
impl<T> GpuVec<T> {
	/// Asserts that the capacity is less than or equal to [`isize::max_value()`](https://doc.rust-lang.org/nightly/std/primitive.isize.html#method.max_value).
	fn assert_capacity_valid(capacity: usize) {
		if capacity > isize::max_value() as usize {
			panic!("invalid capacity");
		}
	}

	/// Constructs a new, empty `GpuVec<T>`. No memory is allocated until needed.
	pub fn new() -> Self {
		Self::with_capacity(0)
	}

	/// Constructs a new, empty `GpuVec<T>` with a specified capacity.
	pub fn with_capacity(capacity: usize) -> Self {
		Self::assert_capacity_valid(capacity);
		if capacity == 0 {
			GpuVec {
				_type: PhantomData,
				ptr: std::ptr::null_mut(),
				len: 0,
				capacity,
			}
		} else {
			let ptr = unsafe { cuda_alloc_device::<T>(capacity).expect("CUDA error") };
			GpuVec {
				_type: PhantomData,
				ptr,
				len: 0,
				capacity,
			}
		}
	}

	/// Creates a `GpuVec<T>` directly from the raw components of another vector.
	/// 
	/// # Safety
	/// `ptr` should be a valid CUDA device pointer, pointing to a memory buffer with a size of at least `capacity * size_of::<T>()` bytes.
	/// 
	/// `ptr` is freed when the `GpuVec` is dropped.
	/// 
	/// # Panics
	/// Panics if `len > capacity`
	pub unsafe fn from_raw_parts(ptr: *mut T, len: usize, capacity: usize) -> Self {
		if len > capacity {
			panic!("len > capacity");
		}
		GpuVec {
			_type: PhantomData,
			ptr,
			len,
			capacity,
		}
	}

	/// Consumes the `GpuVec`, returning the raw pointer, length and capacity.
	/// 
	/// After calling this function, the caller is responsible for cleaning up the memory
	/// previously managed by the `GpuVec`. They are also responsible for calling the destructors
	/// on each `T` when they go out of scope.
	pub unsafe fn into_raw_parts(mut self) -> (*mut T, usize, usize) {
		let len = self.len();
		let capacity = self.capacity();
		let ptr = self.as_mut_ptr();
		self.ptr = ::std::ptr::null_mut();
		self.len = 0;
		self.capacity = 0;
		(ptr, len, capacity)
	}

	/// Try to create a `GpuVec<T>` from a slice of data on the CPU.
	/// 
	/// # Panics
	/// Panics if `len` overflows
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::GpuVec;
	/// let mut v = GpuVec::try_from_slice(&[1, 2, 3][..]).expect("CUDA error");
	/// assert_eq!(v.copy_to_vec(), &[1, 2, 3]);
	/// ```
	pub fn try_from_slice(data: &[T]) -> CudaResult<Self> where T: Copy {
		let len = data.len();
		let len_bytes = len.checked_mul(size_of::<T>()).expect("overflow");
		let capacity = if len_bytes.is_power_of_two() {
			len_bytes
		} else {
			len_bytes.checked_next_power_of_two().unwrap_or(len_bytes)
		};
		unsafe {
			let ptr = cuda_alloc_device::<T>(capacity)?;

			let dst = ptr as *mut T;
			let src = data.as_ptr() as *const T;
			cuda_memcpy(dst, src, len, CudaMemcpyKind::HostToDevice)?;

			Ok(GpuVec {
				_type: PhantomData,
				ptr,
				len,
				capacity,
			})
		}
	}

	/// Tries to moves a vector of values from host to device storage.
	/// 
	/// # Panics
	/// Panics if `len` overflows
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::GpuVec;
	/// let data = vec![vec![1, 2, 3], vec![3, 4, 5]];
	/// 
	/// // Note that `v` is a GPU vector of CPU vectors. The CPU vectors still need to be
	/// // moved to the CPU to be accessed.
	/// let mut v = GpuVec::try_from_vec(data.clone()).expect("CUDA error");
	/// assert_eq!(v.into_vec(), data);
	/// ```
	pub fn try_from_vec(data: Vec<T>) -> CudaResult<Self> {
		let mut v = GpuVec::with_capacity(data.len());
		unsafe {
			cuda_memcpy(v.as_mut_ptr(), data.as_ptr(), data.len(), CudaMemcpyKind::HostToDevice)?;
			v.set_len(data.len());
			// Don't call `Drop` on `T`, just drop the `Vec`
			let _ = std::mem::transmute::<Vec<T>, Vec<ManuallyDrop<T>>>(data);
		}
		Ok(v)
	}

	/// Moves the data in the slice to the CPU and returns it as a `Vec`.
	/// 
	/// This is a relatively slow operation, as it requires moving memory between the RAM and the GPU.
	/// 
	/// # Panics
	/// Panics if a CUDA error is encountered while performing this operation.
	pub fn into_vec(mut self) -> Vec<T> {
		let mut v = Vec::with_capacity(self.len());
		unsafe {
			cuda_memcpy(v.as_mut_ptr(), self.as_ptr(), self.len(), CudaMemcpyKind::DeviceToHost).expect("CUDA error");
			v.set_len(self.len());
			// Set length to 0 to avoid double-freeing
			self.set_len(0);
		}
		v
	}

	/// Returns the number of elements the vector can hold without reallocating.
	pub fn capacity(&self) -> usize {
		self.capacity
	}

	/// Allocates memory such that `additional` extra elements can be pushed to the vector without reallocating.
	/// 
	/// New capacity can be more than or equal to `len + additional`.
	/// 
	/// # Panics
	/// Panics if `len + additional` overflows,
	/// or if a CUDA error is encountered while performing this operation (the most common being an out of memory error).
	/// 
	/// See [`try_reserve()`](#method.try_reserve).
	pub fn reserve(&mut self, additional: usize) {
		self.try_reserve(additional).expect("CUDA error")
	}

	/// Allocates memory such that `additional` extra elements can be pushed to the vector without reallocating.
	/// 
	/// If there is space for `additional` extra elements already, nothing happens.
	/// 
	/// New capacity is otherwise equal to `len + additional`.
	/// 
	/// # Panics
	/// Panics if `len + additional` overflows,
	/// or if a CUDA error is encountered while performing this operation (the most common being an out of memory error).
	/// 
	/// See [`try_reserve()`](#method.try_reserve).
	pub fn reserve_exact(&mut self, additional: usize) {
		self.try_reserve_exact(additional).expect("CUDA error")
	}

	/// Tries to allocate memory such that `additional` extra elements can be pushed to the vector without reallocating.
	/// 
	/// New capacity can be more than or equal to `len + additional`.
	/// 
	/// # Panics
	/// Panics if `len + additional` overflows.
	pub fn try_reserve(&mut self, additional: usize) -> CudaResult<()> {
		let mut new_capacity = self.len.checked_add(additional).expect("overflow");
		Self::assert_capacity_valid(new_capacity);
		if new_capacity <= self.capacity {
			// No need to realloc
			return Ok(());
		}
		if !new_capacity.is_power_of_two() {
			new_capacity = new_capacity.checked_next_power_of_two().unwrap_or(new_capacity);
		}
		self.try_reserve_exact(new_capacity - self.capacity)
	}

	/// Tries to allocate memory such that `additional` extra elements can be pushed to the vector without reallocating.
	/// 
	/// If there is space for `additional` extra elements already, nothing happens.
	/// 
	/// New capacity is otherwise equal to `len + additional`.
	/// 
	/// # Panics
	/// Panics if `len + additional` overflows.
	pub fn try_reserve_exact(&mut self, additional: usize) -> CudaResult<()> {
		let new_capacity = self.len.checked_add(additional).expect("overflow");
		Self::assert_capacity_valid(new_capacity);
		if new_capacity <= self.capacity {
			// No need to realloc
			return Ok(());
		}
		unsafe { cuda_realloc(&mut self.ptr, self.len, new_capacity)? };
		self.capacity = new_capacity;
		Ok(())
	}

	/// Removes all excess capacity of the vector
	/// 
	/// # Panics
	/// Panics if there is a CUDA error while reallocating the vector.
	pub fn shrink_to_fit(&mut self) {
		if self.capacity > self.len {
			unsafe {
				cuda_realloc(&mut self.ptr, self.len, self.len).expect("CUDA error");
				self.capacity = self.len;
			}
		}
	}

	/// Shortens the vector, keeping the first `len` elements and dropping the rest
	pub fn truncate(&mut self, len: usize) {
		if len >= self.len {
			return;
		}
		self.len = len;
		// TODO: Shrink if optimal
	}

	/// Borrows the vector as a slice
	pub fn as_slice<'a>(&'a self) -> &'a GpuSlice<T> {
		self
	}

	/// Borrows the vector as a mutable slice
	pub fn as_slice_mut<'a>(&'a mut self) -> &'a mut GpuSlice<T> {
		self
	}

	/// Sets the length of the vector to `new_len`
	pub unsafe fn set_len(&mut self, len: usize) {
		self.len = len;
	}

	/// Removes an element from the vector and returns it.
	/// 
	/// The removed element is replaced by the last element in the vector.
	/// 
	/// This is a relatively slow operation, as it requires moving memory between the RAM and the GPU.
	/// 
	/// # Panics
	/// Panics if a CUDA error is encountered while performing this operation.
	pub fn swap_remove(&mut self, index: usize) -> T {
		let index_ptr = self.ptr.wrapping_offset(index as isize);
		let end_ptr = self.ptr.wrapping_offset(self.len as isize - 1);
		let mut x = MaybeUninit::<T>::uninit();
		
		unsafe {
			// Copy removed element to `x`
			cuda_memcpy(x.as_mut_ptr(), index_ptr, 1, CudaMemcpyKind::DeviceToHost).expect("CUDA error");
			// Copy end element to removed element's position
			cuda_memcpy(index_ptr, end_ptr, 1, CudaMemcpyKind::DeviceToDevice).expect("CUDA error");
			// Subtract 1 from length of vector
			self.len -= 1;
			x.assume_init()
		}
	}

	/// Inserts an element at position `index` within the vector, shifting all elements after it to the right.
	/// 
	/// Currently this is a relatively slow operation as currently it requires copying the entire vector into a new one.
	/// 
	/// # Panics
	/// Panics if `index` is out of bounds,
	/// or if adding an element to the vector would make the capacity overflow.
	/// 
	/// Also panics if a CUDA error is encountered while performing this operation (the most common being an out of memory error).
	pub fn insert(&mut self, index: usize, element: T) {
		// TODO: Make more efficient somehow by using CUDA kernels
		if index > self.len {
			panic!("index out of bounds");
		}

		// Allocate new vector
		self.reserve(1);
		unsafe {
			let dst = cuda_alloc_device::<T>(self.capacity).expect("CUDA error");
			// Copy first part
			if index > 0 {
				cuda_memcpy(dst, self.ptr, index, CudaMemcpyKind::DeviceToDevice).expect("CUDA error");
			}
			// Copy element
			let element = ManuallyDrop::new(element);
			cuda_memcpy(dst.offset(index as isize), &*element as *const T, 1, CudaMemcpyKind::HostToDevice).expect("CUDA error");
			// Copy last part
			if index < self.len {
				cuda_memcpy(dst.offset(index as isize + 1), self.ptr, self.len - index - 1, CudaMemcpyKind::DeviceToDevice).expect("CUDA error");
			}
			
			// Switch memory buffers
			let old_ptr = self.ptr;
			self.ptr = dst;
			cuda_free_device(old_ptr).expect("CUDA error");
		}
	}

	/// Removes and returns the element at position `index`, shifting all elements after it to the left.
	///
	/// Currently this is a relatively slow operation as currently it requires copying the elements after `index`
	/// into a new vector.
	/// 
	/// # Panics
	/// Panics if `index` is out of bounds,
	/// or if a CUDA error is encountered while performing this operation (the most common being an out of memory error).
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::GpuVec;
	/// let mut v = GpuVec::from(&[1, 2, 3][..]);
	/// assert_eq!(v.remove(1), 2);
	/// assert_eq!(&v.copy_to_vec(), &[1, 3]);
	/// ```
	pub fn remove(&mut self, index: usize) -> T {
		// TODO: Make more efficient somehow by using CUDA kernels
		if index >= self.len {
			panic!("index out of bounds");
		}

		unsafe {
			let mut x = MaybeUninit::<T>::uninit();
			// Copy element at `index`
			cuda_memcpy(x.as_mut_ptr(), self.ptr.add(index), 1, CudaMemcpyKind::DeviceToHost).expect("CUDA error");
			
			if index != self.len - 1 {
				// Copy elements after `index` to new buffer & back again
				let rest_len = self.len - index - 1;
				let tmp = cuda_alloc_device::<T>(rest_len).expect("CUDA error");
				cuda_memcpy(tmp, self.ptr.add(index + 1), rest_len, CudaMemcpyKind::DeviceToDevice).expect("CUDA error");
				cuda_memcpy(self.ptr.add(index), tmp, rest_len, CudaMemcpyKind::DeviceToDevice).expect("CUDA error");
			}
			self.len -= 1;

			x.assume_init()
		}
	}

	//pub fn retain<F>(&mut self, f: F) where F: FnMut(&T) -> bool;
	//pub fn dedup_by_key<F, K>(&mut self, key: F) where F: FnMut(&mut T) -> K, K: PartialEq<K>;
	//pub fn dedup_by<F>(&mut self, same_bucket: F) where F: FnMut(&mut T, &mut T) -> bool;

	/// Appends `value` to the end of the vector.
	/// 
	/// This is a relatively slow operation, as it requires moving memory between the RAM and the GPU.
	/// 
	/// # Panics
	/// Panics if adding an element to the vector would make the capacity overflow,
	/// or if a CUDA error is encountered while performing this operation (the most common being an out of memory error).
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::GpuVec;
	/// let mut v = GpuVec::from(&[1, 2, 3][..]);
	/// v.push(4);
	/// assert_eq!(&v.copy_to_vec(), &[1, 2, 3, 4]);
	/// ```
	pub fn push(&mut self, value: T) {
		self.reserve(1);
		unsafe {
			let value = ManuallyDrop::new(value);
			cuda_memcpy(self.ptr.add(self.len), &*value as *const T, 1, CudaMemcpyKind::HostToDevice).expect("CUDA error");
		}
		self.len += 1;
	}

	/// Pops a value from the end of the vector and returns it, or `None` if it is empty.
	/// 
	/// This is a relatively slow operation, as it requires moving memory between the RAM and the GPU.
	/// 
	/// # Panics
	/// Panics if a CUDA error is encountered while performing this operation
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::GpuVec;
	/// let mut v = GpuVec::from(&[1, 2, 3][..]);
	/// assert_eq!(v.pop(), Some(3));
	/// assert_eq!(v.pop(), Some(2));
	/// assert_eq!(v.pop(), Some(1));
	/// assert_eq!(v.pop(), None);
	/// ```
	pub fn pop(&mut self) -> Option<T> {
		if self.len == 0 {
			None
		} else {
			unsafe {
				self.len -= 1;
				Some(cuda_copy_value_from_device(self.ptr.add(self.len)).expect("CUDA error"))
				// TODO: Shrink if optimal
			}
		}
	}

	/// Appends the elements of another vector onto the end of this one.
	/// 
	///	# Panics
	/// Panics if a CUDA error is encountered while performing this operation (the most common being an out of memory error),
	/// or if adding the elements to this vector would make the capacity overflow.
	pub fn append(&mut self, other: &GpuSlice<T>) {
		self.reserve(other.len());
		unsafe {
			cuda_memcpy(self.ptr, other.as_ptr(), other.len(), CudaMemcpyKind::DeviceToDevice).expect("CUDA error");
		}
		self.len += other.len();
	}

	// TODO: Everything in https://doc.rust-lang.org/std/vec/struct.Vec.html below append()

	/// Deallocates the memory held by the `GpuVec<T>`
	pub fn free(self) -> CudaResult<()> {
		let e = if <T as HasDrop>::has_drop() {
			// `Drop` requires that the data is on the CPU, so copy the data to a `Vec` before dropping
			let mut vec = Vec::with_capacity(self.len());
			unsafe {
				let e = cuda_memcpy(vec.as_mut_ptr(), self.as_ptr(), self.len(), CudaMemcpyKind::DeviceToHost);
				if e.is_ok() {
					vec.set_len(self.len())
				}
				e
			}
		} else {
			Ok(())
		};
		unsafe {
			e.and(cuda_free_device(self.ptr))
		}
	}
}

impl<T: Copy> From<&[T]> for GpuVec<T> {
	fn from(data: &[T]) -> Self {
		Self::try_from_slice(data).expect("CUDA error")
	}
}

// TODO
// impl<T: Copy + Clone> From<&[T]> for GpuVec<T> {
// 	fn from(data: &[T]) -> Self {
// 		Self::try_from_slice(data).expect("CUDA error")
// 	}
// }

impl<T> From<Vec<T>> for GpuVec<T> {
	fn from(data: Vec<T>) -> Self {
		Self::try_from_vec(data).expect("CUDA error")
	}
}

impl<T> Default for GpuVec<T> {
	fn default() -> Self {
		GpuVec::new()
	}
}

impl<T> Deref for GpuVec<T> {
	type Target = GpuSlice<T>;

	fn deref(&self) -> &GpuSlice<T> {
		unsafe { GpuSlice::from_raw_parts(self.ptr, self.len) }
	}
}

impl<T> DerefMut for GpuVec<T> {
	fn deref_mut(&mut self) -> &mut GpuSlice<T> {
		unsafe { GpuSlice::from_raw_parts_mut(self.ptr, self.len) }
	}
}

impl<T: Clone> Clone for GpuVec<T> {
	fn clone(&self) -> Self {
		if <T as HasCopy>::has_copy() {
			// Copy the raw bytes from one to another
			unsafe {
				let mut v = Self::with_capacity(self.len());
				cuda_memcpy(v.as_mut_ptr(), self.as_ptr(), self.len(), CudaMemcpyKind::DeviceToDevice).expect("CUDA error");
				v.set_len(self.len());
				v
			}
		} else {
			// Copy GpuVec to CPU
			let mut v: Vec<ManuallyDrop<T>> = Vec::with_capacity(self.len());
			unsafe {
				cuda_memcpy(v.as_mut_ptr() as *mut T, self.as_ptr() as *const T, self.len(), CudaMemcpyKind::DeviceToHost).expect("CUDA error");
				v.set_len(self.len());
			}
			// Call `clone()` on every element
			let mut ret: Vec<T> = Vec::with_capacity(self.len());
			for elem in &v {
				ret.push(elem.deref().clone());
			}
			// Move to device
			Self::from(ret)
		}
	}
}
trait HasCopy {
	fn has_copy() -> bool;
}
cfg_if! {
	if #[cfg(feature="unstable")] {
		impl<T> HasCopy for T {
			default fn has_copy() -> bool {
				false
			}
		}
		impl<T> HasCopy for T where T: Copy {
			fn has_copy() -> bool {
				true
			}
		}
	} else {
		// Specialization is not enabled -- no way to tell if there is a `Copy` impl on `T`, so assume no types have it
		impl<T> HasCopy for T {
			fn has_copy() -> bool {
				false
			}
		}
	}
}

impl<T> Drop for GpuVec<T> {
	fn drop(&mut self) {
		if <T as HasDrop>::has_drop() {
			// `Drop` requires that the data is on the CPU
			let mut vec = Vec::with_capacity(self.len());
			unsafe {
				if let Ok(_) = cuda_memcpy(vec.as_mut_ptr(), self.as_ptr(), self.len(), CudaMemcpyKind::DeviceToHost) {
					vec.set_len(self.len())
				}
			}
		}
		unsafe {
			cuda_free_device(self.ptr).ok();
		}
	}
}

trait HasDrop {
	fn has_drop() -> bool;
}
cfg_if! {
	if #[cfg(feature="unstable")] {
		impl<T> HasDrop for T {
			default fn has_drop() -> bool {
				false
			}
		}
		impl<T> HasDrop for T where T: Drop {
			fn has_drop() -> bool {
				true
			}
		}
	} else {
		// Specialization is not enabled -- no way to tell if there is a `Drop` impl on `T`, so assume all types have it
		impl<T> HasDrop for T {
			fn has_drop() -> bool {
				true
			}
		}
	}
}

// TODO: impl<T> PartialOrd<Vec<T>> for GpuVec<T> where T: PartialOrd<T>
// TODO: impl<T> Ord for GpuVec<T> where T: Ord

// TODO: impl<T> IntoIterator for GpuVec<T>
// TODO: impl<'a, T> IntoIterator for &'a mut GpuVec<T>
// TODO: impl<'a, T> IntoIterator for &'a GpuVec<T>

impl<T> AsRef<GpuSlice<T>> for GpuVec<T> {
	fn as_ref(&self) -> &GpuSlice<T> {
		self.deref()
	}
}

impl<T> AsMut<GpuSlice<T>> for GpuVec<T> {
	fn as_mut(&mut self) -> &mut GpuSlice<T> {
		self.deref_mut()
	}
}

impl<T> AsRef<GpuVec<T>> for GpuVec<T> {
	fn as_ref(&self) -> &GpuVec<T> {
		self
	}
}

impl<T> AsMut<GpuVec<T>> for GpuVec<T> {
	fn as_mut(&mut self) -> &mut GpuVec<T> {
		self
	}
}

impl<T, I: GpuSliceRange<T>> ops::Index<I> for GpuVec<T> {
	type Output = GpuSlice<T>;
	fn index(&self, index: I) -> &GpuSlice<T> {
		index.slice(self)
	}
}
impl<T, I: GpuSliceRange<T>> ops::IndexMut<I> for GpuVec<T> {
	fn index_mut(&mut self, index: I) -> &mut GpuSlice<T> {
		index.slice_mut(self)
	}
}

// TODO: impl<T> FromIterator<T> for GpuVec<T>


// TODO: Vec ops: https://doc.rust-lang.org/std/vec/struct.Vec.html

#[cfg(test)]
mod tests {
	use super::*;
	
	#[derive(Debug, Clone, PartialEq, Eq)]
	pub struct DropThing<T: std::fmt::Debug + Clone + std::cmp::PartialEq + std::cmp::Eq>(T);
	impl<T> Drop for DropThing<T> where T: std::fmt::Debug + Clone + std::cmp::PartialEq + std::cmp::Eq {
		fn drop(&mut self) {
			println!("Dropping thing: {:?}", self.0);
		}
	}
	
	#[test]
	pub fn test_vec() {
		let data = vec![DropThing(1), DropThing(2), DropThing(3)];
		println!("a");
		let v = GpuVec::from(data.clone());
		println!("b");
		let v = v.into_vec();
		println!("c");
		assert_eq!(v, data);
		println!("d");
	}
	
	#[test]
	pub fn test_slice() {
		let data: GpuVec<_> = vec![1, 2, 3, 4, 5].into();
		assert_eq!((&data[..]).copy_to_vec(), &[1, 2, 3, 4, 5]);
		assert_eq!((&data[0..]).copy_to_vec(), &[1, 2, 3, 4, 5]);
		assert_eq!((&data[1..]).copy_to_vec(), &[2, 3, 4, 5]);
		assert_eq!((&data[..4]).copy_to_vec(), &[1, 2, 3, 4]);
		assert_eq!((&data[..5]).copy_to_vec(), &[1, 2, 3, 4, 5]);
		assert_eq!((&data[1..2]).copy_to_vec(), &[2]);
		assert_eq!((&data[..=2]).copy_to_vec(), &[1, 2, 3]);
		assert_eq!((&data[1..=2]).copy_to_vec(), &[2, 3]);
		assert_eq!((&(&data[1..4])[1..2]).copy_to_vec(), &[3]);
	}
	
	#[test]
	pub fn test_into_vec() {
		let data = vec![vec![1, 2, 3], vec![3, 4, 5]];
		let v = GpuVec::try_from_vec(data.clone()).expect("CUDA error");
		assert_eq!(v.into_vec(), data);
	}
}
