use cuda::runtime::*;
use cuda_macros::*;

/// Callable from CPU, run on CPU
#[host]
unsafe fn host(ptr: *mut i32) {
    *ptr = 1;
}

/// Callable from GPU, run on GPU
#[device]
unsafe fn device(ptr: *mut i32) {
    *ptr = 1;
}

/// Callable from CPU & GPU, run on GPU
#[global]
unsafe fn global(ptr: *mut i32) {
    *ptr = 1;
}

/// Callable from CPU & GPU, run on device which it is called from
#[host]
#[device]
unsafe fn host_device(ptr: *mut i32) {
    *ptr = 1;
}

/// Callable from CPU & GPU, run on device which it is called from
///
/// There shouldn't be any difference between this and `host_device`, but we keep this in to
/// check if both ways work
#[device]
#[host]
unsafe fn device_host(ptr: *mut i32) {
    *ptr = 1;
}

#[global]
unsafe fn call_device(ptr: *mut i32) {
    device(ptr);
}

#[global]
unsafe fn call_host_device(ptr: *mut i32) {
    host_device(ptr);
}

#[global]
unsafe fn call_device_host(ptr: *mut i32) {
    device_host(ptr);
}

fn setup_cuda_test() -> CudaResult<*mut i32> {
    let ptr = cuda_alloc_device(std::mem::size_of::<i32>())?;
    unsafe {
        cuda_memset(ptr, 0, std::mem::size_of::<i32>())?;
    }
    return Ok(ptr as _);
}

unsafe fn teardown_cuda_test(ptr: *mut i32) -> CudaResult<i32> {
    let mut i = 0;
    cuda_memcpy(&mut i as _, ptr, 1, CudaMemcpyKind::DeviceToHost)?;
    CudaDevice::synchronize_current()?;
    return Ok(i);
}

#[test]
fn test_host() {
    let mut i = 0;
    unsafe {
        host(&mut i as *mut i32);
    }
    assert_eq!(i, 1);
}

#[test]
fn test_device() -> CudaResult {
    let ptr = setup_cuda_test()?;
    unsafe {
        call_device(ptr);
        assert_eq!(teardown_cuda_test(ptr)?, 1);
    }
    Ok(())
}

#[test]
fn test_global() -> CudaResult {
    let ptr = setup_cuda_test()?;
    unsafe {
        global(ptr);
        assert_eq!(teardown_cuda_test(ptr)?, 1);
    }
    Ok(())
}

#[test]
fn test_host_device_cpu() {
    let mut i = 0;
    unsafe {
        host_device(&mut i as *mut i32);
    }
    assert_eq!(i, 1);
}

#[test]
fn test_host_device_gpu() -> CudaResult {
    let ptr = setup_cuda_test()?;
    unsafe {
        call_host_device(ptr);
        assert_eq!(teardown_cuda_test(ptr)?, 1);
    }
    Ok(())
}

#[test]
fn test_device_host_cpu() {
    let mut i = 0;
    unsafe {
        device_host(&mut i as *mut i32);
    }
    assert_eq!(i, 1);
}

#[test]
fn test_device_host_gpu() -> CudaResult {
    let ptr = setup_cuda_test()?;
    unsafe {
        call_device_host(ptr);
        assert_eq!(teardown_cuda_test(ptr)?, 1);
    }
    Ok(())
}