
use std::cell::RefCell;

use crate::*;


thread_local! {
	pub static TEMP_DEVICE_STORAGE: RefCell<GpuVec<u8>> = RefCell::new(GpuVec::new());
}


pub unsafe fn reverse_vector<T>(vec: *mut T, len: usize) {
	let elem_size = std::mem::size_of::<T>();
	let conf = ExecutionConfig::from_num_threads_with_shared_mem_per_thread(len as u32 / 2, elem_size);
	global_reverse_vector(conf, vec as *mut u8, elem_size as u32, len as u32);
}

#[global]
unsafe fn global_reverse_vector(vec: *mut u8, elem_size: u32, len: u32) {
	#[shared]
	let tmp: [u8];
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
		global_contains(conf, tmp.as_mut_ptr() as *mut bool, x, vec, len);
		*tmp.get(0).unwrap() != 0
	})
}

#[global]
pub unsafe fn global_contains<T>(found: *mut bool, x: T, vec: *const T, len: usize) where T: GpuType {
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
		global_eq(conf, tmp.as_mut_ptr() as *mut bool, lhs, rhs, len);
		*tmp.get(0).unwrap() == 0
	})
}

#[global]
pub unsafe fn global_eq<T>(ret_neq: *mut bool, lhs: *const T, rhs: *const T, len: usize) where T: GpuType {
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
