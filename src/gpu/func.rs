
use std::cell::RefCell;
use std::convert::TryInto;
use std::mem::size_of;

use crate::*;


thread_local! {
	pub static TEMP_DEVICE_STORAGE: RefCell<GpuVec<u8>> = RefCell::new(GpuVec::new());
}


pub unsafe fn swap<T>(a: *mut T, b: *mut T) {
	global_swap((1, 1, size_of::<T>()), a as *mut u8, b as *mut u8, size_of::<T>().try_into().unwrap());
}

#[global]
unsafe fn global_swap(a: *mut u8, b: *mut u8, elem_size: u32) {
	#[shared]
	let mut tmp: [u8];
	for i in 0..elem_size {
		tmp[i] = a[i];
	}
	for i in 0..elem_size {
		a[i] = b[i];
	}
	for i in 0..elem_size {
		b[i] = tmp[i];
	}
}

pub unsafe fn reverse_vector<T>(vec: *mut T, len: usize) {
	let elem_size = size_of::<T>();
	let conf = ExecutionConfig::from_num_threads_with_shared_mem_per_thread(len as u32 / 2, elem_size);
	global_reverse_vector(conf, vec as *mut u8, elem_size as u32, len as u32);
}

#[global]
unsafe fn global_reverse_vector(vec: *mut u8, elem_size: u32, len: u32) {
	#[shared]
	let mut tmp: [u8];
	let i: u32 = blockDim.x * blockIdx.x + threadIdx.x;
	let ri: u32 = len - 1 - i;

	let byte_i: u32 = i * elem_size;
	let byte_ri: u32 = ri * elem_size;

	let byte_shared_i: u32 = threadIdx.x * elem_size;

	if i < len / 2 {
		//printf("%d,%d,%d - %d,%d,%d (/%d,%d,%d): %d <-> %d\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, i, ri);
		for j in 0..elem_size {
			tmp[byte_shared_i + j] = vec[byte_i + j];
		}
		for j in 0..elem_size {
			vec[byte_i + j] = vec[byte_ri + j];
		}
		for j in 0..elem_size {
			vec[byte_ri + j] = tmp[byte_shared_i + j];
		}
	}
}

pub unsafe fn contains<T>(x: T, vec: *const T, len: usize) -> bool where T: GpuType {
	TEMP_DEVICE_STORAGE.with(|tmp| {
		let mut tmp = tmp.borrow_mut();
		tmp.clear();
		tmp.push(0);
		let conf = ExecutionConfig::from_num_threads(len as u32);
		global_contains(conf, tmp.as_mut_ptr() as *mut bool, x, vec, len as u32);
		*tmp.get(0).unwrap() != 0
	})
}

#[global]
pub unsafe fn global_contains<T>(found: *mut bool, x: T, vec: *const T, len: u32) where T: GpuType {
	let i: u32 = blockDim.x * blockIdx.x + threadIdx.x;
	if i < len && vec[i] == x {
		*found = true;
	}
}

pub unsafe fn eq<T>(lhs: *const T, rhs: *const T, len: usize) -> bool where T: GpuType {
	TEMP_DEVICE_STORAGE.with(|tmp| {
		let mut tmp = tmp.borrow_mut();
		tmp.clear();
		tmp.push(0);
		let conf = ExecutionConfig::from_num_threads(len as u32);
		global_eq(conf, tmp.as_mut_ptr() as *mut bool, lhs, rhs, len as u32);
		*tmp.get(0).unwrap() == 0
	})
}

#[global]
pub unsafe fn global_eq<T>(ret_neq: *mut bool, lhs: *const T, rhs: *const T, len: u32) where T: GpuType {
	let i: u32 = blockDim.x * blockIdx.x + threadIdx.x;
	if i < len && lhs[i] != rhs[i] {
		*ret_neq = true;
	}
}

pub unsafe fn ne<T>(lhs: *const T, rhs: *const T, len: usize) -> bool where T: GpuType {
	!eq(lhs, rhs, len)
}

// #[global]
// pub unsafe fn global_ne<T>(lhs: *const T, rhs: *const T, len: usize) where T: GpuType {
	
// }

pub unsafe fn swap_slice<T>(a: *mut T, b: *mut T, len: usize) {
	let len = len as u32;
	let conf = ExecutionConfig::from_num_threads_with_shared_mem_per_thread(len, size_of::<T>());
	if size_of::<T>() == size_of::<u8>() {
		global_swap_slice_generic(conf, a as *mut u8, b as *mut u8, len);
	} else if size_of::<T>() == size_of::<u16>() {
		global_swap_slice_generic(conf, a as *mut u16, b as *mut u16, len);
	} else if size_of::<T>() == size_of::<u32>() {
		global_swap_slice_generic(conf, a as *mut u32, b as *mut u32, len);
	} else if size_of::<T>() == size_of::<u64>() {
		global_swap_slice_generic(conf, a as *mut u64, b as *mut u64, len);
	} else {
		global_swap_slice(conf, a as *mut u8, b as *mut u8, size_of::<T>() as u32, len);
	}
}

#[global]
unsafe fn global_swap_slice(a: *mut u8, b: *mut u8, elem_size: u32, len: u32) {
	#[shared]
	let mut tmp: [u8];
	let i: u32 = blockDim.x * blockIdx.x + threadIdx.x;
	if i < len {
		for j in 0..elem_size {
			tmp[threadIdx.x + j] = a[i + j];
		}
		for j in 0..elem_size {
			a[i + j] = b[i + j];
		}
		for j in 0..elem_size {
			b[i + j] = tmp[threadIdx.x + j];
		}
	}
}

#[global]
unsafe fn global_swap_slice_generic<T: GpuType>(a: *mut T, b: *mut T, len: u32) {
	#[shared]
	let mut tmp: [T];
	let i: u32 = blockDim.x * blockIdx.x + threadIdx.x;
	if i < len {
		tmp[threadIdx.x] = a[i];
		a[i] = b[i];
		b[i] = tmp[threadIdx.x];
	}
}

/// Returns `(gpu_vec, dim, strides)`
pub fn gpu_vec_from_ndarray<T, S, D>(array: nd::ArrayBase::<S, D>) -> (GpuVec<T>, D, D) where T: GpuType, S: nd::Data<Elem=T>, D: nd::Dimension {
	let shape = array.shape();
	let v = if let Some(s) = array.as_slice() {
		GpuVec::from(s)
	} else {
		GpuVec::from(&array.iter().map(|&x| x).collect::<Vec<_>>())
	};

	// Compute default array strides
	// e.g. shape (a, b, c) => strides (b * c, c, 1)
	let mut strides = D::zeros(array.ndim());
	if shape.iter().all(|&d| d != 0) {
		let mut it = strides.slice_mut().iter_mut().rev();
		// Set first element to 1
		if let Some(rs) = it.next() {
			*rs = 1;
		}
		let mut cum_prod = 1;
		for (rs, dim) in it.zip(shape.iter().rev()) {
			cum_prod *= *dim;
			*rs = cum_prod;
		}
	}
	
	(v, array.raw_dim(), strides)
}
