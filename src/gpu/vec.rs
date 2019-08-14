
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::mem::{MaybeUninit, size_of};
use std::fmt;

use crate::rcuda::*;
use super::func;


/// Mutable reference to a device location
/// 
/// Values that are written to this are cached until this is dropped, or the [flush()](#method.flush) is called, at which point
/// the value is written to the device
#[derive(Debug)]
pub struct GpuMutRef<'a, T: Copy> {
	_phantom: PhantomData<&'a T>,
	ptr: *mut T,
	value: T,
}
unsafe impl<'a, T: Copy + Send> Send for GpuMutRef<'a, T> {}
impl<'a, T: Copy> GpuMutRef<'a, T> {
	/// Constructs a new `GpuMutRef<T>`.
	/// 
	/// # Safety
	/// `ptr` must be a valid device pointer that lasts for the lifetime `'a`.
	pub unsafe fn new(ptr: *mut T, value: T) -> Self {
		GpuMutRef {
			_phantom: PhantomData,
			ptr,
			value,
		}
	}

	/// Writes the value given into the cache
	pub fn write(&mut self, value: T) {
		self.value = value;
	}

	/// Copies the cached value to the pointer location given
	pub fn flush(mut self) -> CudaResult<()> {
		let ptr = self.ptr;
		self.ptr = std::ptr::null_mut();
		if ptr != std::ptr::null_mut() {
			unsafe {
				cuda_copy_value_to_device(ptr, self.value)
			}
		} else {
			Ok(())
		}
	}
}
impl<'a, T: Copy> Deref for GpuMutRef<'a, T> {
	type Target = T;
	fn deref(&self) -> &T {
		&self.value
	}
}
impl<'a, T: Copy> DerefMut for GpuMutRef<'a, T> {
	fn deref_mut(&mut self) -> &mut T {
		&mut self.value
	}
}
impl<'a, T: Copy> Drop for GpuMutRef<'a, T> {
	/// Flushes the value to the device.
	fn drop(&mut self) {
		if self.ptr != std::ptr::null_mut() {
			unsafe {
				cuda_copy_value_to_device(self.ptr, self.value).ok();
			}
		}
	}
}


/// Slice of [`GpuVec`](struct.GpuVec.html).
/// 
/// [`Unsize`](https://doc.rust-lang.org/std/marker/trait.Unsize.html)d type that can be only accessed via a reference (e.g. `&mut GpuSlice`).
/// 
/// Can be created by dereferencing a GpuVec, see [`GpuVec::deref()`](struct.GpuVec.html#method.deref).
pub struct GpuSlice<T: Copy>([T]);
impl<T: Copy> GpuSlice<T> {
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
	/// If there is a CUDA error while performing the operation.
	pub fn to_vec(&self) -> Vec<T> {
		let mut v = Vec::with_capacity(self.len());
		unsafe {
			cuda_memcpy(v.as_mut_ptr(), self.as_ptr(), self.len(), CudaMemcpyKind::DeviceToHost).expect("CUDA error");
			v.set_len(self.len());
		}
		v
	}

	/// Copies the data in the slice to a new `GpuVec`.
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::*;
	/// let a = GpuVec::new(&[1, 2, 3, 4][..]);
	/// let b: GpuVec<_> = &a[1..].to_gpu_vec();
	/// assert_eq!(b.to_vec(), &[2, 3, 4]);
	/// ```
	/// 
	/// # Panics
	/// If there is a CUDA error while performing the operation.
	pub fn to_gpu_vec(&self) -> GpuVec<T> {
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
	/// If there is a CUDA error while performing the operation.
	pub fn first(&self) -> Option<T> {
		if self.len() == 0 {
			None
		} else {
			unsafe {
				Some(cuda_copy_value_from_device(self.as_ptr()).expect("CUDA error"))
			}
		}
	}

	/// Gets a mutable reference to the first element in the slice
	/// 
	/// This is a relatively slow operation, as it requires moving memory between the RAM and the GPU.
	/// 
	/// # Panics
	/// If there is a CUDA error while performing the operation.
	pub fn first_mut<'a>(&'a mut self) -> Option<GpuMutRef<'a, T>> {
		if self.len() == 0 {
			None
		} else {
			unsafe {
				let ptr = self.as_mut_ptr();
				let value = cuda_copy_value_from_device(ptr).expect("CUDA error");
				Some(GpuMutRef::new(ptr, value))
			}
		}
	}

	/// Gets the last element in the slice
	/// 
	/// This is a relatively slow operation, as it requires moving memory between the RAM and the GPU.
	/// 
	/// # Panics
	/// If there is a CUDA error while performing the operation.
	pub fn last(&self) -> Option<T> {
		if self.len() == 0 {
			None
		} else {
			unsafe {
				Some(cuda_copy_value_from_device(self.0.as_ptr().add(self.len() - 1)).expect("CUDA error"))
			}
		}
	}

	/// Gets a mutable reference to the last element in the slice
	/// 
	/// This is a relatively slow operation, as it requires moving memory between the RAM and the GPU.
	/// 
	/// # Panics
	/// If there is a CUDA error while performing the operation.
	pub fn last_mut<'a>(&'a mut self) -> Option<GpuMutRef<'a, T>> {
		if self.len() == 0 {
			None
		} else {
			unsafe {
				let ptr = self.as_mut_ptr().add(self.len() - 1);
				let value = cuda_copy_value_from_device(ptr).expect("CUDA error");
				Some(GpuMutRef::new(ptr, value))
			}
		}
	}

	//pub fn get()
	//pub fn get_mut()

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
	/// # Examples
	/// ```
	/// # use cuda_util::*;
	/// let mut v = GpuVec::from(&[1, 2, 3][..]);
	/// v.swap(0, 1);
	/// assert_eq!(v.to_vec(), &[2, 1, 3]);
	/// ```
	/// 
	/// # Panics
	/// - If there is a CUDA error while performing the operation.
	/// - If `a` or `b` are out of bounds.
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

	/// Reverses the order of the slice.
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::*;
	/// let mut cpu_vec: Vec<u32> = (0..10_000).into_iter().collect();
	/// let mut v = GpuVec::from(&cpu_vec[..]);
	/// v.reverse();
	/// cpu_vec.reverse();
	/// assert_eq!(v.to_vec(), &cpu_vec);
	/// ```
	pub fn reverse(&mut self) {
		func::reverse_vector(self.as_mut_ptr(), self.len());
	}
	// TODO: Everything in https://doc.rust-lang.org/std/primitive.slice.html below reverse()
}
impl<T: Copy> fmt::Debug for GpuSlice<T> {
	fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
		write!(f, "GpuSlice([<{:p}>; {}])", self.as_ptr(), self.len())?;
		Ok(())
	}
}
// TODO: Slice ops: https://doc.rust-lang.org/std/primitive.slice.html

/// Contiguous growable array type, similar to `Vec<T>` but stored on the GPU.
#[derive(Debug)]
pub struct GpuVec<T: Copy> {
	_type: PhantomData<T>,
	ptr: *mut T,
	len: usize,
	capacity: usize,
}
impl<T: Copy> GpuVec<T> {
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
	/// If `len > capacity`
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

	/// Try to create a `GpuVec<T>` from a slice of data on the CPU.
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::GpuVec;
	/// let mut v = GpuVec::try_from_slice(&[1, 2, 3][..]).expect("CUDA error");
	/// assert_eq!(v.to_vec(), &[1, 2, 3]);
	/// ```
	/// 
	/// # Panics
	/// If `len` overflows
	pub fn try_from_slice(data: &[T]) -> CudaResult<Self> {
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

	/// Returns the number of elements the vector can hold without reallocating.
	pub fn capacity(&self) -> usize {
		self.capacity
	}

	/// Allocates memory such that `additional` extra elements can be pushed to the vector without reallocating.
	/// 
	/// New capacity can be more than or equal to `len + additional`.
	/// 
	/// # Panics
	/// - If there is a CUDA error while performing the operation (the most common being an out of memory error).
	/// See [`try_reserve()`](#method.try_reserve).
	/// - If `len + additional` overflows an `isize`.
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
	/// - If there is a CUDA error while performing the operation (the most common being an out of memory error).
	/// See [`try_reserve_exact()`](#method.try_reserve_exact).
	/// - If `len + additional` overflows an `isize`.
	pub fn reserve_exact(&mut self, additional: usize) {
		self.try_reserve_exact(additional).expect("CUDA error")
	}

	/// Tries to allocate memory such that `additional` extra elements can be pushed to the vector without reallocating.
	/// 
	/// New capacity can be more than or equal to `len + additional`.
	/// 
	/// # Panics
	/// If `len + additional` overflows an `isize`.
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
	/// If `len + additional` overflows an `isize`.
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
	/// If there is a CUDA error while reallocating the vector.
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
	/// If there is a CUDA error while performing the operation.
	pub fn swap_remove(&mut self, index: usize) -> T {
		let index_ptr = self.ptr.wrapping_offset(index as isize);
		let end_ptr = self.ptr.wrapping_offset(self.len as isize - 1);
		let mut x = MaybeUninit::<T>::uninit();
		
		unsafe {
			// Copy removed element to `x`
			cuda_memcpy(x.as_mut_ptr(), index_ptr, 1, CudaMemcpyKind::DeviceToHost).expect("CUDA error");
			// Copy end elemnt to removed element's position
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
	/// - If there is a CUDA error while performing the operation (the most common being an out of memory error).
	/// - If `index` is out of bounds.
	/// - If adding an element to the vector would make the capacity overflow.
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
			cuda_memcpy(dst.offset(index as isize), &element as *const T, 1, CudaMemcpyKind::HostToDevice).expect("CUDA error");
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
	/// # Examples
	/// ```
	/// # use cuda_util::GpuVec;
	/// let mut v = GpuVec::from(&[1, 2, 3][..]);
	/// assert_eq!(v.remove(1), 2);
	/// assert_eq!(&v.to_vec(), &[1, 3]);
	/// ```
	/// 
	/// # Panics
	/// - If there is a CUDA error while performing the operation (the most common being an out of memory error).
	/// - If `index` is out of bounds.
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
	/// # Examples
	/// ```
	/// # use cuda_util::GpuVec;
	/// let mut v = GpuVec::from(&[1, 2, 3][..]);
	/// v.push(4);
	/// assert_eq!(&v.to_vec(), &[1, 2, 3, 4]);
	/// ```
	/// 
	/// # Panics
	/// - If there is a CUDA error while performing the operation (the most common being an out of memory error).
	/// - If adding an element to the vector would make the capacity overflow.
	pub fn push(&mut self, value: T) {
		self.reserve(1);
		unsafe {
			cuda_memcpy(self.ptr.add(self.len), &value as *const T, 1, CudaMemcpyKind::HostToDevice).expect("CUDA error");
		}
		self.len += 1;
	}

	/// Pops a value from the end of the vector and returns it, or `None` if it is empty.
	/// 
	/// This is a relatively slow operation, as it requires moving memory between the RAM and the GPU.
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
	/// 
	/// # Panics
	/// If there is a CUDA error while performing the operation
	pub fn pop(&mut self) -> Option<T> {
		if self.len == 0 {
			None
		} else {
			unsafe {
				self.len -= 1;
				Some(cuda_copy_value_from_device(self.ptr.add(self.len)).expect("CUDA error"))
			}
		}
	}

	/// Appends the elements of another vector onto the end of this one.
	/// 
	///	# Panics
	/// - If there is a CUDA error while performing the operation (the most common being an out of memory error).
	/// - If adding the elements to this vector would make the capacity overflow.
	pub fn append(&mut self, other: &GpuSlice<T>) {
		self.reserve(other.len());
		unsafe {
			cuda_memcpy(self.ptr, other.as_ptr(), other.len(), CudaMemcpyKind::DeviceToDevice).expect("CUDA error");
		}
		self.len += other.len();
	}

	// TODO: Everything in https://doc.rust-lang.org/std/vec/struct.Vec.html below append()

	/// Deallocates the memory held by the `GpuVec<T>`
	pub fn free(mut self) -> CudaResult<()> {
		let ptr = self.ptr;
		self.ptr = std::ptr::null_mut();
		self.capacity = 0;
		self.len = 0;
		unsafe {
			cuda_free_device(ptr)
		}
	}
}
impl<T: Copy> From<&[T]> for GpuVec<T> {
	fn from(data: &[T]) -> Self {
		Self::try_from_slice(data).expect("CUDA error")
	}
}

impl<T: Copy> Deref for GpuVec<T> {
	type Target = GpuSlice<T>;

	fn deref(&self) -> &GpuSlice<T> {
		unsafe { GpuSlice::from_raw_parts(self.ptr, self.len) }
	}
}

impl<T: Copy> DerefMut for GpuVec<T> {
	fn deref_mut(&mut self) -> &mut GpuSlice<T> {
		unsafe { GpuSlice::from_raw_parts_mut(self.ptr, self.len) }
	}
}

impl<T: Copy> Clone for GpuVec<T> {
	fn clone(&self) -> Self {
		unsafe {
			let mut v = Self::with_capacity(self.len());
			cuda_memcpy(v.as_mut_ptr(), self.as_ptr(), self.len(), CudaMemcpyKind::DeviceToDevice).expect("CUDA error");
			v.set_len(self.len());
			v
		}
	}
}

impl<T: Copy> Drop for GpuVec<T> {
	fn drop(&mut self) {
		unsafe { cuda_free_device(self.ptr).ok() };
	}
}

// TODO: Vec ops: https://doc.rust-lang.org/std/vec/struct.Vec.html
