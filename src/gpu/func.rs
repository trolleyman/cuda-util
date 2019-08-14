
use crate::*;


fn div_ceil(x: usize, divisor: usize) -> usize {
	x / divisor + (x % divisor != 0) as usize
}

pub fn reverse_vector<T: CudaNumber>(vec: *mut T, len: usize) {
	let num_threads_total = len / 2;
	let num_threads_per_block = 1024;
	let num_blocks = div_ceil(num_threads_total, num_threads_per_block);
	global_reverse_vector((num_threads_per_block, num_blocks), vec, len);
}

#[global]
fn global_reverse_vector<T: CudaNumber>(vec: *mut T, len: usize) {
	let i: u32 = blockDim.x * blockIdx.x + threadIdx.x;
	if i < len / 2 {
		let x: T = vec[i];
		vec[i] = vec[len - i];
		vec[len - i] = x;
	}
}
