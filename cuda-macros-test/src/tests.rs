
use cuda::runtime::*;

use cuda_macros::*;

#[device]
unsafe fn device1() -> i32 {
	2
}

#[device]
unsafe fn device2() -> i32 {
	return 2;
}

#[global]
unsafe fn call_device1(x: *mut i32) {
	*x = device1();
}

#[global]
unsafe fn call_device2(x: *mut i32) {
	*x = device2();
}

#[test]
fn test_device() {
	let mut y = 0;
	unsafe {
		let x = cuda_alloc_device(std::mem::size_of::<i32>()).unwrap() as *mut i32;
		// device1
		cuda_memset(x as *mut u8, 1, std::mem::size_of::<i32>()).unwrap();
		call_device1((32, 1), x);
		cuda_memcpy(&mut y, x, 1, CudaMemcpyKind::DeviceToHost).unwrap();
		assert_eq!(y, 2);
		
		// device2
		cuda_memset(x as *mut u8, 1, std::mem::size_of::<i32>()).unwrap();
		call_device2((32, 1), x);
		cuda_memcpy(&mut y, x, 1, CudaMemcpyKind::DeviceToHost).unwrap();
		assert_eq!(y, 2);
		cuda_free_device(x as *mut u8).unwrap();
	}
	CudaDevice::synchronize_current()
		.expect("CUDA deviceSynchronize failed");
}

#[host]
#[device]
unsafe fn host_device(x: *mut i32) -> i32 {
	*x = 2;
	return 3;
}

#[device]
#[host]
unsafe fn device_host(x: *mut i32) -> i32 {
	*x = 2;
	return 3;
}

#[test]
fn test_host_device_on_host() {
	let mut x = 0;
	assert_eq!(unsafe { host_device(&mut x) }, 3);
	assert_eq!(x, 2);
	
	x = 0;
	assert_eq!(unsafe { device_host(&mut x) }, 3);
	assert_eq!(x, 2);
}

#[global]
unsafe fn call_host_device(x: *mut i32, ret: *mut i32) {
	*ret = host_device(x);
}

#[global]
unsafe fn call_device_host(x: *mut i32, ret: *mut i32) {
	*ret = device_host(x);
}

#[test]
fn test_host_device_on_device() {
	let mut x = 0;
	let mut ret = 0;
	unsafe {
		let x_device = cuda_alloc_device(std::mem::size_of::<i32>()).unwrap() as *mut i32;
		let ret_device = cuda_alloc_device(std::mem::size_of::<i32>()).unwrap() as *mut i32;
		// host_device
		cuda_memset(x_device as *mut u8, 1, std::mem::size_of::<i32>()).unwrap();
		cuda_memset(ret_device as *mut u8, 1, std::mem::size_of::<i32>()).unwrap();
		call_host_device((1, 1), x_device, ret_device);
		cuda_memcpy(&mut x, x_device, 1, CudaMemcpyKind::DeviceToHost).unwrap();
		cuda_memcpy(&mut ret, ret_device, 1, CudaMemcpyKind::DeviceToHost).unwrap();
		assert_eq!(x, 2);
		assert_eq!(ret, 3);
		
		// device_host
		cuda_memset(x_device as *mut u8, 1, std::mem::size_of::<i32>()).unwrap();
		cuda_memset(ret_device as *mut u8, 1, std::mem::size_of::<i32>()).unwrap();
		call_device_host((1, 1), x_device, ret_device);
		cuda_memcpy(&mut x, x_device, 1, CudaMemcpyKind::DeviceToHost).unwrap();
		cuda_memcpy(&mut ret, ret_device, 1, CudaMemcpyKind::DeviceToHost).unwrap();
		assert_eq!(x, 2);
		assert_eq!(ret, 3);
	}
	CudaDevice::synchronize_current()
		.expect("CUDA deviceSynchronize failed");
}

#[global]
unsafe fn map_add(data: *mut i32, x: i32) {
	data[blockDim.x * blockIdx.x + threadIdx.x] += x;
}

#[test]
fn test_add() {
	let num_blocks = 8;
	let num_threads = 256;
	let total_threads = num_blocks * num_threads;
	let num = 12;

	let data: Vec<_> = (0..total_threads as i32).into_iter().collect();
	let mut data_copy = vec![0i32; total_threads];
	unsafe {
		let data_device = cuda_alloc_device(total_threads * std::mem::size_of::<i32>()).unwrap() as *mut i32;
		cuda_memcpy(data_device, data.as_ptr(), total_threads, CudaMemcpyKind::HostToDevice).unwrap();
		map_add((num_threads as u32, num_blocks as u32), data_device, num);
		cuda_memcpy(data_copy.as_mut_ptr(), data_device, total_threads, CudaMemcpyKind::DeviceToHost).unwrap();
	}
	println!("{:?}", data);
	println!("{:?}", data_copy);
	for (&x, &y) in data.iter().zip(data_copy.iter()) {
		assert_eq!(x + num, y);
	}
}
