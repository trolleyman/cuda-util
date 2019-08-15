
use crate::*;


fn div_ceil(x: usize, divisor: usize) -> usize {
	x / divisor + (x % divisor != 0) as usize
}

pub fn reverse_vector<T: Copy>(vec: *mut T, len: usize) {
	let num_threads_total = len / 2;
	let num_threads_per_block = 1024;
	let num_blocks = div_ceil(num_threads_total, num_threads_per_block);
	let num_threads_per_block = if num_blocks == 1 { num_threads_total } else { num_threads_per_block };
	let elem_size = std::mem::size_of::<T>();
	unsafe {
		let conf = ExecutionConfig::from((num_blocks as u32, num_threads_per_block as u32, num_threads_per_block * elem_size));
		global_reverse_vector(conf, vec as *mut u8, elem_size as u32, len as u32);
	}
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
		printf("%d,%d,%d - %d,%d,%d (/%d,%d,%d): %d <-> %d\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, i, ri);
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
