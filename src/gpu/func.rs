
use crate::*;


fn div_ceil(x: usize, divisor: usize) -> usize {
	x / divisor + (x % divisor != 0) as usize
}

pub fn reverse_vector<T: Copy>(vec: *mut T, len: usize) {
	let num_threads_total = len / 2;
	let num_threads_per_block = 1024;
	let num_blocks = div_ceil(num_threads_total, num_threads_per_block);
	let elem_size = std::mem::size_of::<T>();
	global_reverse_vector((num_threads_per_block as u32, num_blocks as u32, elem_size as u32), vec as *mut u8, elem_size, len);
}

#[global]
unsafe fn global_reverse_vector(vec: *mut u8, elem_size: usize, len: usize) {
	#[shared]
	let elem: [u8];
	let i: usize = (blockDim.x * blockIdx.x + threadIdx.x) * elem_size;
	if i < len / 2 {
		let x: T = vec[i];
		vec[i] = vec[len - i];
		vec[len - i] = x;
	}
}
 