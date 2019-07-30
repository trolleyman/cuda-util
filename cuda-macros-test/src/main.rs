use cuda_macros::*;

use cuda::runtime::*;

#[global]
unsafe fn hello(x: *mut i32, y: i32) {
	printf("Hello from block %d, thread %d (y=%d)\n", blockIdx.x, threadIdx.x, y);
	*x = 2;
}


#[test]
fn test_hello() {
	let x = cuda_alloc_device(std::mem::size_of::<i32>()).unwrap() as *mut i32;
	let mut y = 0;
	unsafe {
		cuda_memset(x as *mut u8, 1, std::mem::size_of::<i32>()).unwrap();
		hello((32, 1), x, 1);
		cuda_memcpy(&mut y, x, 1, CudaMemcpyKind::DeviceToHost).unwrap();
	}
	CudaDevice::synchronize_current()
		.expect("CUDA deviceSynchronize failed");
	assert_eq!(y, 2);
}
