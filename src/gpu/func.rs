
use lazy_static::lazy_static;

use crate::*;


lazy_static!{
	static ref TEMP_DEVICE_STORAGE: GpuVec<u8> = GpuVec::new();
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
	TEMP_DEVICE_STORAGE.clear();
	TEMP_DEVICE_STORAGE.push(0);
	let conf = ExecutionConfig::from_num_threads(len as u32);
	global_contains(conf, TEMP_DEVICE_STORAGE.as_mut_ptr() as *mut bool, x, vec, len);
	*TEMP_DEVICE_STORAGE.get(0).unwrap() != 0
}

#[global]
pub unsafe fn global_contains<T>(found: *mut bool, x: T, vec: *const T, len: usize) where T: GpuType {
	let i: u32 = blockDim.x * blockIdx.x + threadIdx.x;
	if (vec[i] == x) {
		*found = true;
	}
}

pub unsafe fn eq<T>(lhs: *const T, lhs_len: usize, rhs: *const T, rhs_len: usize) -> bool where T: GpuType {
	unimplemented!()
}

// #[global]
// pub unsafe fn global_eq<T>(lhs: *const T, lhs_len: usize, rhs: *const T, rhs_len: usize) where T: GpuType {
	
// }

pub unsafe fn ne<T>(lhs: *const T, lhs_len: usize, rhs: *const T, rhs_len: usize) -> bool where T: GpuType {
	unimplemented!()
}

// #[global]
// pub unsafe fn global_ne<T>(lhs: *const T, lhs_len: usize, rhs: *const T, rhs_len: usize) where T: GpuType {
	
// }
