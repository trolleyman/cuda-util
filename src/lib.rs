
extern crate proc_macro;
extern crate proc_macro2;
extern crate quote;
extern crate syn;

use proc_macro2::{Span, TokenStream};


fn error_on_attrs(attr: TokenStream) -> proc_macro::TokenStream {
    let mut it = attr.into_iter();
    let mut span = it.next().map(|tt| tt.span()).unwrap_or(Span::def_site());
    span = it.fold(span, |s| s.join(s).unwrap_or(s));
    syn::Error::new(span, "no options taken").to_compile_error().into()
}


#[proc_macro_attribute]
pub fn host(attr: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let attr = TokenStream::from(attr);
    let item = TokenStream::from(item);
    if !attr.is_empty() {
        error_on_attrs(attr);
    }

    let output: TokenStream = TokenStream::new();

    proc_macro::TokenStream::from(output)
}

#[proc_macro_attribute]
pub fn device(attr: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let attr = TokenStream::from(attr);
    let item = TokenStream::from(item);
    if !attr.is_empty() {
        error_on_attrs(attr);
    }

    let output: TokenStream = TokenStream::new();

    proc_macro::TokenStream::from(output)
}

#[proc_macro_attribute]
pub fn global(attr: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let attr = TokenStream::from(attr);
    let item = TokenStream::from(item);
    if !attr.is_empty() {
        error_on_attrs(attr);
    }

    let output: TokenStream = TokenStream::new();

    proc_macro::TokenStream::from(output)
}


#[cfg(test)]
mod tests {
    use cuda::runtime::*;

    /// callable from CPU, run on CPU
    #[host]
    unsafe fn host(ptr: *mut i32) {
        *ptr = 1;
    }
    
    /// callable from GPU, run on GPU
    #[device]
    unsafe fn device(ptr: *mut i32) {
        *ptr = 1;
    }
    
    /// callable from CPU & GPU, run on GPU
    #[global]
    unsafe fn global(ptr: *mut i32) {
        *ptr = 1;
    }
    
    /// callable from CPU & GPU, run on device which it is called from
    #[host]
    #[device]
    unsafe fn host_device(ptr: *mut i32) {
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
    
    fn setup_cuda_test() -> CudaResult<*mut i32> {
        let ptr = cuda_alloc_device(std::mem::size_of::<i32>())?;
        cuda_memset(ptr, 0, std::mem::size_of::<i32>())?;
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
        let mut ptr = setup_cuda_test()?;
        unsafe {
            call_device(ptr);
            assert_eq!(teardown_cuda_test(ptr)?, 1);
        }
        Ok(())
    }
    
    #[test]
    fn test_global() -> CudaResult {
        let mut ptr = setup_cuda_test()?;
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
        let mut ptr = setup_cuda_test()?;
        unsafe {
            call_host_device(ptr);
            assert_eq!(teardown_cuda_test(ptr)?, 1);
        }
        Ok(())
    }
}
