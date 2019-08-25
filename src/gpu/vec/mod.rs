
use std::marker::PhantomData;
use std::ops::{self, Deref, DerefMut};
use std::cmp;
use std::mem::{ManuallyDrop, MaybeUninit, size_of};
use std::fmt;
use std::convert::{AsRef, AsMut};

use crate::GpuType;
use crate::rcuda::*;
use super::func;

mod index;
pub use self::index::*;
pub mod reference;
use self::reference::*;
mod type_util;
pub use self::type_util::CopyIfStable;


/// Creates a [`GpuVec`](struct.GpuVec.html) from the arguments.
/// 
/// Has the same syntax as [`std::vec!`](https://doc.rust-lang.org/std/macro.vec.html).
#[macro_export]
macro_rules! gvec {
    ($elem:expr; $n:expr) => (
        $crate::GpuVec::from(vec![$elem; $n])
    );
    ($($x:expr),*) => (
		$crate::GpuVec::from(vec![$($x),*])
    );
    ($($x:expr,)*) => ($crate::gvec![$($x),*])
}


/// Slice of [`GpuVec`](struct.GpuVec.html).
/// 
/// [`Unsize`](https://doc.rust-lang.org/std/marker/trait.Unsize.html)d type that can be only accessed via a reference
/// (e.g. `&mut GpuSlice`).
/// 
/// Can be created by dereferencing a GpuVec, see [`GpuVec::deref()`](struct.GpuVec.html#method.deref).
/// 
/// `T` is bound by the [`CopyIfStable`](trait.CopyIfStable.html) trait. See the [`GpuVec`](struct.GpuVec.html)
/// documentation for details.
pub struct GpuSlice<T: CopyIfStable>(ManuallyDrop<[T]>);
impl<T: CopyIfStable> GpuSlice<T> {
	/// Constructs an immutable GpuSlice from raw parts.
	/// 
	/// # Safety
	/// `data` must point to a device buffer of at least size `len` that lives for at least lifetime `'a`.
	pub unsafe fn from_raw_parts<'a>(data: *const T, len: usize) -> &'a Self {
		std::mem::transmute(std::slice::from_raw_parts(data, len))
	}

	/// Constructs a mutable GPU slice from raw parts.
	/// 
	/// # Safety
	/// `data` must point to a device buffer of at least size `len` that lives for at least lifetime `'a`.
	pub unsafe fn from_raw_parts_mut<'a>(data: *mut T, len: usize) -> &'a mut Self {
		std::mem::transmute(std::slice::from_raw_parts_mut(data, len))
	}

	/// Copies the data in the slice to the host and returns it as a vector.
	#[cfg_attr(feature="unstable", doc="\nIf `T: !Copy`, `clone()` is called for each element.")]
	/// 
	/// This is a relatively slow operation, as it requires moving memory between the RAM and the GPU.
	/// 
	/// # Panics
	/// Panics if a CUDA error occurs.
	pub fn to_vec(&self) -> Vec<T> where T: Clone {
		if type_util::has_copy::<T>() {
			// If `T: Copy`, then we don't need to call `clone()` for each element
			let mut v = Vec::with_capacity(self.len());
			unsafe {
				cuda_memcpy(v.as_mut_ptr(), self.as_ptr(), self.len(), CudaMemcpyKind::DeviceToHost).expect("CUDA error");
				v.set_len(self.len());
			}
			v
		} else {
			self.clone_to_vec()
		}
	}

	fn clone_to_vec(&self) -> Vec<T> where T: Clone {
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

	/// Copies the data in the slice to the CPU and returns it as a `GpuSliceRef`.
	/// The lifetime of this reference is linked to the lifetime of the `GpuSlice`.
	/// 
	/// This is a relatively slow operation, as it requires moving memory between the RAM and the GPU.
	/// 
	/// # Panics
	/// Panics if a CUDA error occurs.
	pub fn borrow_as_cpu_slice<'a>(&'a self) -> GpuSliceRef<'a, T> {
		unsafe {
			GpuSliceRef::from_device_ptr(self.as_ptr(), self.len()).expect("CUDA error")
		}
	}

	/// Copies the data in the slice to the CPU and returns it as a mutable `GpuSliceRef`.
	/// The lifetime of this reference is linked to the lifetime of the `GpuSlice`.
	/// 
	/// This is a relatively slow operation, as it requires moving memory between the RAM and the GPU.
	/// 
	/// # Panics
	/// Panics if a CUDA error occurs.
	pub fn borrow_as_cpu_slice_mut<'a>(&'a mut self) -> GpuSliceMutRef<'a, T> {
		unsafe {
			GpuSliceMutRef::from_device_ptr(self.as_mut_ptr(), self.len()).expect("CUDA error")
		}
	}

	/// Copies the data in the slice to a new `GpuVec`.
	/// 
	/// # Panics
	/// Panics if a CUDA error occurs.
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::prelude::*;
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
	/// Panics if a CUDA error occurs.
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
	/// Panics if a CUDA error occurs.
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
	/// Panics if a CUDA error occurs.
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
	/// Panics if a CUDA error occurs.
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
	/// # use cuda_util::prelude::*;
	/// let mut v = GpuVec::from(&[1, 2, 100, 4, 5][..]);
	/// assert_eq!(*v.get(2).unwrap(), 100);
	/// ```
	/// 
	/// # Panics
	/// Panics if a CUDA error occurs. See [`try_get()`](#method.try_get).
	pub fn get<'a>(&'a self, index: usize) -> Option<GpuRef<'a, T>> {
		self.try_get(index).expect("CUDA error")
	}

	/// Returns a mutable reference to the element at position `index`, or `None` if out of bounds.
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::prelude::*;
	/// let mut v = GpuVec::from(&[1, 2, 3, 4, 5][..]);
	/// {
	/// 	let mut r = v.get_mut(2).unwrap();
	/// 	*r = 100;
	/// }
	/// assert_eq!(v.to_vec(), &[1, 2, 100, 4, 5]);
	/// ```
	/// 
	/// # Panics
	/// Panics if a CUDA error occurs. See [`try_get_mut()`](#method.try_get_mut).
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
	/// or if a CUDA error occurs.
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::prelude::*;
	/// let mut v = GpuVec::from(&[1, 2, 3][..]);
	/// v.swap(0, 1);
	/// assert_eq!(v.to_vec(), &[2, 1, 3]);
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
			func::swap(a_ptr, b_ptr);
		}
	}

	/// Reverses the order of the elements in the slice.
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::prelude::*;
	/// let mut cpu_vec: Vec<u32> = (0..2000).into_iter().collect();
	/// let mut v = GpuVec::from(&cpu_vec[..]);
	/// v.reverse();
	/// cpu_vec.reverse();
	/// assert_eq!(&v.to_vec(), &cpu_vec);
	/// ```
	pub fn reverse(&mut self) {
		unsafe {
			func::reverse_vector(self.as_mut_ptr(), self.len());
		}
	}

	// TODO: iter(&self)
	// TODO: iter_mut(&mut self)

	// TODO: windows(&self, size: usize)

	// TODO: chunks(&self, chunk_size: usize)
	// TODO: chunks_mut(&mut self, chunk_size: usize)
	// TODO: chunks_exact(&self, chunk_size: usize)
	// TODO: chunks_exact_mut(&mut self, chunk_size: usize)

	// TODO: rchunks(&self, chunk_size: usize)
	// TODO: rchunks_mut(&mut self, chunk_size: usize)
	// TODO: rchunks_exact(&self, chunk_size: usize)
	// TODO: rchunks_exact_mut(&mut self, chunk_size: usize)

	// TODO: split_at(&self, mid: usize) -> (&GpuSlice<T>, &GpuSlice<T>)
	// TODO: split_at_mut(&self, mid: usize) -> (&mut GpuSlice<T>, &mut GpuSlice<T>)

	// TODO: split, split_mut, rsplit, rsplit_mut, splitn, splitn_mut, rsplitn, rsplitn_mut

	/// Returns `true` if the slice contains an element with a given value.
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::prelude::*;
	/// let v = GpuVec::from(&[5, 1, 3][..]);
	/// assert!(v.contains(&5));
	/// assert!(v.contains(&1));
	/// assert!(v.contains(&3));
	/// assert!(!v.contains(&0));
	/// ```
	/// 
	/// ```
	/// # use cuda_util::prelude::*;
	/// let v = GpuVec::from(&[0.0, 1.0, std::f32::NAN][..]);
	/// assert!(v.contains(&0.0));
	/// assert!(v.contains(&1.0));
	/// assert!(!v.contains(&std::f32::NAN));  // NaN != NaN
	/// ```
	pub fn contains(&self, x: &T) -> bool where T: GpuType {
		unsafe {
			func::contains(*x, self.as_ptr(), self.len())
		}
	}

	/// Returns `true` if `needle` is a prefix of the slice.
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::prelude::*;
	/// let v = gvec![1, 2, 3, 4, 5];
	/// assert!(v.starts_with(&gvec![1, 2, 3]));
	/// assert!(!v.starts_with(&gvec![1, 4, 5]));
	/// ```
	pub fn starts_with(&self, needle: &GpuSlice<T>) -> bool where T: GpuType {
		if self.len() < needle.len() {
			false
		} else {
			&self[..needle.len()] == needle
		}
	}

	/// Returns `true` if `needle` is a suffix of the slice
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::prelude::*;
	/// let v = gvec![1, 2, 3, 4, 5];
	/// assert!(v.ends_with(&gvec![4, 5]));
	/// assert!(!v.ends_with(&gvec![1, 4, 5]));
	/// ```
	pub fn ends_with(&self, needle: &GpuSlice<T>) -> bool where T: GpuType {
		if self.len() < needle.len() {
			false
		} else {
			&self[self.len() - needle.len()..] == needle
		}
	}

	// TODO: rotate_left, rotate_right
	// TODO: copy_within

	/// Swaps all elements in `self` with those in `other`.
	/// 
	/// The length of `other` must be the same as `self`.
	/// 
	/// # Panics
	/// Panics if the two slices have different lengths
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::prelude::*;
	/// let mut v1 = gvec![1, 2, 3, 4];
	/// let mut v2 = gvec![0, 0];
	/// v1[1..3].swap_with_slice(&mut v2);
	/// assert_eq!(v1.to_vec(), [1, 0, 0, 4]);
	/// assert_eq!(v2.to_vec(), [2, 3]);
	/// ```
	pub fn swap_with_slice(&mut self, other: &mut GpuSlice<T>) {
		if self.len() != other.len() {
			panic!("lengths are different: {} != {}", self.len(), other.len());
		}
		unsafe {
			func::swap_slice(self.as_mut_ptr(), other.as_mut_ptr(), self.len());
		}
	}
}

impl<T: CopyIfStable> fmt::Debug for GpuSlice<T> {
	fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
		write!(f, "GpuSlice([<{:p}>; {}])", self.as_ptr(), self.len())
	}
}

impl<T: CopyIfStable, I: GpuSliceRange<T>> ops::Index<I> for GpuSlice<T> {
	type Output = GpuSlice<T>;
	fn index(&self, index: I) -> &GpuSlice<T> {
		index.slice(self)
	}
}
impl<T: CopyIfStable, I: GpuSliceRange<T>> ops::IndexMut<I> for GpuSlice<T> {
	fn index_mut(&mut self, index: I) -> &mut GpuSlice<T> {
		index.slice_mut(self)
	}
}
impl<T: CopyIfStable> cmp::PartialEq<GpuSlice<T>> for GpuSlice<T> where T: GpuType {
	fn eq(&self, other: &GpuSlice<T>) -> bool {
		if self.len() != other.len() {
			false
		} else {
			unsafe {
				func::eq(self.as_ptr(), other.as_ptr(), self.len())
			}
		}
	}
	fn ne(&self, other: &GpuSlice<T>) -> bool {
		if self.len() != other.len() {
			true
		} else {
			unsafe {
				func::ne(self.as_ptr(), other.as_ptr(), self.len())
			}
		}
	}
}
impl<T: CopyIfStable> cmp::Eq for GpuSlice<T> where T: GpuType + cmp::Eq {}

// TODO: Slice ops: https://doc.rust-lang.org/std/primitive.slice.html

impl<T: CopyIfStable> Into<Vec<T>> for &GpuSlice<T> where T: Clone {
	fn into(self) -> Vec<T> {
		self.to_vec()
	}
}


/// Contiguous growable array type, similar to [`Vec<T>`](https://doc.rust-lang.org/std/vec/struct.Vec.html) but stored on the GPU.
/// Pronounced 'GPU vector'.
/// 
/// `T` is bound by the [`CopyIfStable`](trait.CopyIfStable) trait. This is so that `T: Copy` by default,
/// but unbounded when the `unstable` feature is activated. The `unstable` feature is automatically enabled
/// when a nightly compiler is detected.
/// 
/// Specialization ([issue #31844](https://github.com/rust-lang/rust/issues/31844)) is needed for `T` to be unbounded, as
/// otherwise the `drop()` function is hugely inefficient. Without it the library must assume that every type
/// has a `Drop` bound, and is forced to move it to the CPU before dropping it. Since this happens for *every* `GpuVec`,
/// this is a large performance penalty.
pub struct GpuVec<T: CopyIfStable> {
	_type: PhantomData<T>,
	ptr: *mut T,
	len: usize,
	capacity: usize,
}
unsafe impl<T: CopyIfStable + Send> Send for GpuVec<T> {}
unsafe impl<T: CopyIfStable + Sync> Sync for GpuVec<T> {}
impl<T: CopyIfStable> GpuVec<T> {
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
	/// # use cuda_util::prelude::*;
	/// let mut v = GpuVec::try_from_slice(&[1, 2, 3][..]).expect("CUDA error");
	/// assert_eq!(v.to_vec(), &[1, 2, 3]);
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
	/// # use cuda_util::prelude::*;
	/// let data = vec![1, 2, 3, 4];
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
	/// Panics if a CUDA error occurs.
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
	/// or if a CUDA error occurs (the most common being an out of memory error).
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
	/// or if a CUDA error occurs (the most common being an out of memory error).
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
	/// Panics if a CUDA error occurs.
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
	/// Also panics if a CUDA error occurs (the most common being an out of memory error).
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
			let mut old_ptr = self.ptr;
			self.ptr = dst;
			self.len += 1;
			cuda_free_device(&mut old_ptr).expect("CUDA error");
		}
	}

	/// Removes and returns the element at position `index`, shifting all elements after it to the left.
	///
	/// Currently this is a relatively slow operation as currently it requires copying the elements after `index`
	/// into a new vector.
	/// 
	/// # Panics
	/// Panics if `index` is out of bounds,
	/// or if a CUDA error occurs (the most common being an out of memory error).
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::prelude::*;
	/// let mut v = GpuVec::from(&[1, 2, 3][..]);
	/// assert_eq!(v.remove(1), 2);
	/// assert_eq!(&v.to_vec(), &[1, 3]);
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
	/// or if a CUDA error occurs (the most common being an out of memory error).
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::prelude::*;
	/// let mut v = GpuVec::from(&[1, 2, 3][..]);
	/// v.push(4);
	/// assert_eq!(&v.to_vec(), &[1, 2, 3, 4]);
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
	/// Panics if a CUDA error occurs
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::prelude::*;
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
	/// Panics if a CUDA error occurs (the most common being an out of memory error),
	/// or if adding the elements to this vector would make the capacity overflow.
	pub fn append(&mut self, other: &GpuSlice<T>) {
		self.reserve(other.len());
		unsafe {
			cuda_memcpy(self.ptr, other.as_ptr(), other.len(), CudaMemcpyKind::DeviceToDevice).expect("CUDA error");
		}
		self.len += other.len();
	}

	// TODO: drain

	/// Remove all values in this vector.
	///
	/// This has no effect on the allocated capacity of the vector.
	pub fn clear(&mut self) {
		self.try_clear().ok();
	}

	/// Tries to remove all values in this vector. Returns a CUDA error if it was encountered.
	///
	/// This has no effect on the allocated capacity of the vector.
	pub fn try_clear(&mut self) -> CudaResult<()> {
		if type_util::has_drop::<T>() {
			// `Drop` requires that the data is on the CPU, so copy the data to a `Vec` before dropping
			let mut vec = Vec::with_capacity(self.len());
			unsafe {
				let e = cuda_memcpy(vec.as_mut_ptr(), self.as_ptr(), self.len(), CudaMemcpyKind::DeviceToHost);
				if e.is_ok() {
					vec.set_len(self.len())
				}
				self.len = 0;
				e
			}
		} else {
			self.len = 0;
			Ok(())
		}
	}

	
	/// Returns the number of elements in the GPU vector.
	pub fn len(&self) -> usize {
		self.len
	}

	/// Returns `true` if the GPU vector contains no elements.
	pub fn is_empty(&self) -> bool {
		self.len == 0
	}
	
	/// Splits the collection at the given index.
	/// 
	/// Returns a newly allocated GPU vector. `self` contains elements `[0, at)`, and the returned GPU vector contains
	/// the elements `[at, len)`.
	/// 
	/// Note that the capacity of `self` does not change.
	/// 
	/// # Panics
	/// Panics if `at > len`, or if a CUDA error occurs (the most common being an out of memory error).
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::prelude::*;
	/// let mut vec = gvec![1, 2, 3];
	/// let vec2 = vec.split_off(1);
	/// assert_eq!(vec.to_vec(), [1]);
	/// assert_eq!(vec2.to_vec(), [2, 3]);
	/// ```
	pub fn split_off(&mut self, at: usize) -> GpuVec<T> {
		if at > self.len() {
			panic!("index {} out of bounds in GPU vector of length {}", at, self.len());
		}
		let ret_len = self.len() - at;
		let mut ret = GpuVec::with_capacity(ret_len);
		unsafe {
			cuda_memcpy(ret.as_mut_ptr(), self.as_mut_ptr().add(at), ret_len, CudaMemcpyKind::DeviceToDevice).expect("CUDA error");
			ret.set_len(ret_len);
			self.set_len(at);
		}
		ret
	}

	// TODO(device_fn_ptr): resize_with

	/// Resize the GPU vector in-place, so that `len` is equal to `new_len`.
	/// 
	/// If `new_len` is greater than `len`, the GPU vector is extended by the difference, with
	/// each additional slot filled by `value`.
	#[cfg_attr(feature="unstable", doc="If `T: !Copy`, `clone()` is called for each new element.")]
	/// 
	/// If `new_len` is less than `len`, then the GPU vector is truncated.
	/// 
	/// # Panics
	/// Panics if a CUDA error occurs (the most common being an out of memory error).
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::prelude::*;
	/// let mut gv = gvec![1, 2, 3];
	/// gv.resize(4*1024, 4);
	/// assert_eq!(gv[..3].to_vec(), [1, 2, 3]);
	/// assert_eq!(gv[3..].to_vec(), vec![4; 4*1024-3]);
	/// ```
	pub fn resize(&mut self, new_len: usize, value: T) where T: Clone {
		if self.len() == new_len {
			// Do nothing
		} else if self.len() > new_len {
			// Truncate vector
			// Drop some elements from the GPU vector
			if type_util::has_drop::<T>() {
				// Copy to host vector before dropping
				let drop_elements_len = self.len() - new_len;
				let mut v = Vec::with_capacity(drop_elements_len);
				unsafe {
					cuda_memcpy(v.as_mut_ptr(), self.as_ptr(), drop_elements_len, CudaMemcpyKind::DeviceToHost).expect("CUDA error");
					v.set_len(drop_elements_len);
					self.set_len(new_len);
				}
			} else {
				unsafe {
					self.set_len(new_len);
				}
			}
		} else {
			// Extend vector
			let extra_len = new_len - self.len();
			self.reserve(extra_len);
			if type_util::has_copy::<T>() {
				let mut chunk_len = 1024.min(extra_len);
				let end_ptr = unsafe { self.as_mut_ptr().add(self.len()) };
				
				// Copy initial 1024 elements
				let mut extra = Vec::with_capacity(chunk_len);
				extra.resize(chunk_len, value);
				unsafe {
					cuda_memcpy(end_ptr, extra.as_ptr(), chunk_len, CudaMemcpyKind::HostToDevice).expect("CUDA error");
				}
				
				// Fill GPU vector using data that has already been copied to the GPU
				let mut num_copied = chunk_len;
				while num_copied < extra_len {
					let copy_len = chunk_len.min(extra_len - num_copied);
					unsafe {
						cuda_memcpy(end_ptr.add(num_copied), end_ptr, copy_len, CudaMemcpyKind::DeviceToDevice).expect("CUDA error");
					}
					num_copied += copy_len;
					chunk_len *= 2;
				}
				unsafe {
					self.set_len(new_len);
				}
			} else {
				let mut extra: Vec<ManuallyDrop<T>> = Vec::with_capacity(extra_len);
				extra.resize(extra_len, ManuallyDrop::new(value));
				
				// Copy to GPU vec
				unsafe {
					cuda_memcpy(self.as_mut_ptr().add(self.len()), extra.as_ptr() as *const T, extra_len, CudaMemcpyKind::HostToDevice).expect("CUDA error");
					self.set_len(new_len);
				}
			}
		}
	}

	/// Clones and appends all elements in a slice to the GPU vector.
	//#[cfg_attr(feature="unstable", doc="\nIf `T: !Copy`, this is slow as `other` has to be copied to RAM,")]
	//#[cfg_attr(feature="unstable", doc="`clone` has to be called on each element, and then each element gets copied back again.")]
	/// 
	/// # Panics
	/// Panics if a CUDA error occurs (the most common being an out of memory error).
	/// 
	/// # Examples
	/// ```
	/// # use cuda_util::prelude::*;
	/// let mut v = gvec![1, 2, 3];
	/// let v2 = gvec![4, 5, 6, 7];
	/// v.extend_from_slice(&v2[2..]);
	/// v.extend_from_slice(&v2[..2]);
	/// v.extend_from_slice(&v2[..]);
	/// v.extend_from_slice(&v2[1..2]);
	/// assert_eq!(v.to_vec(), vec![1, 2, 3, 6, 7, 4, 5, 4, 5, 6, 7, 5]);
	/// ```
	pub fn extend_from_slice(&mut self, other: &GpuSlice<T>) where T: Copy {
		self.reserve(other.len());
		unsafe {
			cuda_memcpy(self.as_mut_ptr().add(self.len()), other.as_ptr(), other.len(), CudaMemcpyKind::DeviceToDevice).expect("CUDA error");
			self.set_len(self.len().checked_add(other.len()).unwrap());
		}
	}

	// TODO: Everything in https://doc.rust-lang.org/std/vec/struct.Vec.html below clear()

	/// Deallocates the memory held by the `GpuVec<T>`
	pub fn free(mut self) -> CudaResult<()> {
		unsafe {
			self.try_clear().and(cuda_free_device(&mut self.ptr))
		}
	}
}

impl<T: CopyIfStable> fmt::Debug for GpuVec<T> {
	fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
		write!(f, "GpuVec([<{:p}>; {}])", self.as_ptr(), self.len())
	}
}

impl<T: CopyIfStable + Copy + Clone> From<&[T]> for GpuVec<T> {
	fn from(data: &[T]) -> Self {
		Self::try_from_slice(data).expect("CUDA error")
	}
}

// TODO
// impl<T: CopyIfStable + Clone> From<&[T]> for GpuVec<T> {
// 	default fn from(data: &[T]) -> Self {
// 		Self::try_from_slice(data).expect("CUDA error")
// 	}
// }

impl<T: CopyIfStable> From<&Vec<T>> for GpuVec<T> where T: Copy {
	fn from(data: &Vec<T>) -> Self {
		Self::from(data.as_slice())
	}
}

impl<T: CopyIfStable> From<Vec<T>> for GpuVec<T> {
	fn from(data: Vec<T>) -> Self {
		Self::try_from_vec(data).expect("CUDA error")
	}
}

impl<T: CopyIfStable> Default for GpuVec<T> {
	fn default() -> Self {
		GpuVec::new()
	}
}

impl<T: CopyIfStable> Deref for GpuVec<T> {
	type Target = GpuSlice<T>;

	fn deref(&self) -> &GpuSlice<T> {
		unsafe { GpuSlice::from_raw_parts(self.ptr, self.len) }
	}
}

impl<T: CopyIfStable> DerefMut for GpuVec<T> {
	fn deref_mut(&mut self) -> &mut GpuSlice<T> {
		unsafe { GpuSlice::from_raw_parts_mut(self.ptr, self.len) }
	}
}

impl<T: CopyIfStable + Clone> Clone for GpuVec<T> {
	fn clone(&self) -> Self {
		if type_util::has_copy::<T>() {
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

impl<T: CopyIfStable> Drop for GpuVec<T> {
	fn drop(&mut self) {
		self.clear();
		unsafe {
			cuda_free_device(&mut self.ptr).ok();
		}
	}
}

// TODO: impl<T> PartialOrd<Vec<T>> for GpuVec<T> where T: PartialOrd<T>
// TODO: impl<T> Ord for GpuVec<T> where T: Ord

// TODO: impl<T> IntoIterator for GpuVec<T>
// TODO: impl<'a, T> IntoIterator for &'a mut GpuVec<T>
// TODO: impl<'a, T> IntoIterator for &'a GpuVec<T>

impl<T: CopyIfStable> AsRef<GpuSlice<T>> for GpuVec<T> {
	fn as_ref(&self) -> &GpuSlice<T> {
		self.deref()
	}
}

impl<T: CopyIfStable> AsMut<GpuSlice<T>> for GpuVec<T> {
	fn as_mut(&mut self) -> &mut GpuSlice<T> {
		self.deref_mut()
	}
}

impl<T: CopyIfStable> AsRef<GpuVec<T>> for GpuVec<T> {
	fn as_ref(&self) -> &GpuVec<T> {
		self
	}
}

impl<T: CopyIfStable> AsMut<GpuVec<T>> for GpuVec<T> {
	fn as_mut(&mut self) -> &mut GpuVec<T> {
		self
	}
}

impl<T: CopyIfStable, I: GpuSliceRange<T>> ops::Index<I> for GpuVec<T> {
	type Output = GpuSlice<T>;
	fn index(&self, index: I) -> &GpuSlice<T> {
		index.slice(self)
	}
}
impl<T: CopyIfStable, I: GpuSliceRange<T>> ops::IndexMut<I> for GpuVec<T> {
	fn index_mut(&mut self, index: I) -> &mut GpuSlice<T> {
		index.slice_mut(self)
	}
}

// TODO: impl<T> FromIterator<T> for GpuVec<T>


// TODO: Vec ops: https://doc.rust-lang.org/std/vec/struct.Vec.html

impl<T: CopyIfStable> Into<Vec<T>> for GpuVec<T> {
	fn into(self) -> Vec<T> {
		self.into_vec()
	}
}
impl<T: CopyIfStable> Into<Vec<T>> for &GpuVec<T> where T: Clone {
	fn into(self) -> Vec<T> {
		self.to_vec()
	}
}

impl<T: CopyIfStable> cmp::PartialEq<GpuVec<T>> for GpuSlice<T> where T: GpuType {
	fn eq(&self, other: &GpuVec<T>) -> bool { self == other.as_slice() }
	fn ne(&self, other: &GpuVec<T>) -> bool { self != other.as_slice() }
}
impl<T: CopyIfStable> cmp::PartialEq<&GpuVec<T>> for GpuSlice<T> where T: GpuType {
	fn eq(&self, other: &&GpuVec<T>) -> bool { self == other.as_slice() }
	fn ne(&self, other: &&GpuVec<T>) -> bool { self != other.as_slice() }
}
impl<T: CopyIfStable> cmp::PartialEq<GpuVec<T>> for &GpuSlice<T> where T: GpuType {
	fn eq(&self, other: &GpuVec<T>) -> bool { *self == other.as_slice() }
	fn ne(&self, other: &GpuVec<T>) -> bool { *self != other.as_slice() }
}

impl<T: CopyIfStable> cmp::PartialEq<GpuSlice<T>> for GpuVec<T> where T: GpuType {
	fn eq(&self, other: &GpuSlice<T>) -> bool { self.as_slice() == other }
	fn ne(&self, other: &GpuSlice<T>) -> bool { self.as_slice() != other }
}
impl<T: CopyIfStable> cmp::PartialEq<&GpuSlice<T>> for GpuVec<T> where T: GpuType {
	fn eq(&self, other: &&GpuSlice<T>) -> bool { self.as_slice() == *other }
	fn ne(&self, other: &&GpuSlice<T>) -> bool { self.as_slice() != *other }
}
impl<T: CopyIfStable> cmp::PartialEq<GpuSlice<T>> for &GpuVec<T> where T: GpuType {
	fn eq(&self, other: &GpuSlice<T>) -> bool { self.as_slice() == other }
	fn ne(&self, other: &GpuSlice<T>) -> bool { self.as_slice() != other }
}

impl<T: CopyIfStable> cmp::PartialEq<GpuVec<T>> for GpuVec<T> where T: GpuType {
	fn eq(&self, other: &GpuVec<T>) -> bool { self.as_slice() == other.as_slice() }
	fn ne(&self, other: &GpuVec<T>) -> bool { self.as_slice() != other.as_slice() }
}
impl<T: CopyIfStable> cmp::PartialEq<&GpuVec<T>> for GpuVec<T> where T: GpuType {
	fn eq(&self, other: &&GpuVec<T>) -> bool { self.as_slice() == other.as_slice() }
	fn ne(&self, other: &&GpuVec<T>) -> bool { self.as_slice() != other.as_slice() }
}
impl<T: CopyIfStable> cmp::PartialEq<GpuVec<T>> for &GpuVec<T> where T: GpuType {
	fn eq(&self, other: &GpuVec<T>) -> bool { self.as_slice() == other.as_slice() }
	fn ne(&self, other: &GpuVec<T>) -> bool { self.as_slice() != other.as_slice() }
}

impl<T: CopyIfStable> cmp::Eq for GpuVec<T> where T: GpuType + cmp::Eq {}


#[cfg(test)]
mod tests {
	use super::*;
	
	#[cfg(feature="unstable")]
	#[test]
	pub fn test_vec() {
		#[derive(Debug, Clone, PartialEq, Eq)]
		pub struct DropThing(i32, Option<*mut i32>);
		impl Drop for DropThing {
			fn drop(&mut self) {
				println!("Dropping thing: *{:p} = {:?}", self.1.unwrap_or(::std::ptr::null_mut()), self.0);
				if let Some(ptr) = self.1 {
					unsafe {
						*ptr = self.0;
					}
				}
			}
		}
		
		let mut ret = 0i32;
		{
			let data = vec![DropThing(1, Some(&mut ret as *mut i32)), DropThing(2, None), DropThing(3, None)];
			println!("a");
			let v = GpuVec::from(data.clone());
			println!("b");
			let v = v.into_vec();
			println!("c");
			assert_eq!(v, data);
			println!("d");
		}
		assert_eq!(ret, 1);
	}
	
	#[test]
	pub fn test_slice() {
		let data: GpuVec<_> = vec![1, 2, 3, 4, 5].into();
		assert_eq!((&data[..]).to_vec(), &[1, 2, 3, 4, 5]);
		assert_eq!((&data[0..]).to_vec(), &[1, 2, 3, 4, 5]);
		assert_eq!((&data[1..]).to_vec(), &[2, 3, 4, 5]);
		assert_eq!((&data[..4]).to_vec(), &[1, 2, 3, 4]);
		assert_eq!((&data[..5]).to_vec(), &[1, 2, 3, 4, 5]);
		assert_eq!((&data[1..2]).to_vec(), &[2]);
		assert_eq!((&data[..=2]).to_vec(), &[1, 2, 3]);
		assert_eq!((&data[1..=2]).to_vec(), &[2, 3]);
		assert_eq!((&(&data[1..4])[1..2]).to_vec(), &[3]);
	}
	
	#[cfg(feature="unstable")]
	#[test]
	pub fn test_into_vec() {
		let data = vec![vec![1, 2, 3], vec![3, 4, 5]];
		let v = GpuVec::try_from_vec(data.clone()).expect("CUDA error");
		assert_eq!(v.into_vec(), data);
	}
	
	#[test]
	pub fn test_printing() {
		let mut v = GpuVec::from(&[1, 2, 3][..]);
		println!("{:?}", v);
		println!("{:?}", v.as_slice());
		println!("{:?}", v.as_slice_mut());
		println!("{:?}", v.as_ptr());
		println!("{:?}", v.as_mut_ptr());
		println!("{:?}", v.get(1));
		println!("{:?}", v.get_mut(2));
		println!("{:?}", (&v[1..]).borrow_as_cpu_slice());
		println!("{:?}", (&mut v[1..]).borrow_as_cpu_slice_mut());
	}
	
	#[test]
	pub fn test_eq() {
		let v: Vec<_> = (0..1_000_000).collect();
		assert_eq!(GpuVec::from(&v), GpuVec::from(&v));
		let mut z = GpuVec::from(&v);
		*z.get_mut(500_000).unwrap() = 123;
		assert_ne!(&z, GpuVec::from(&v));
		assert!(z != GpuVec::from(&v));
		println!("{:?}", v.len());
	}
}
