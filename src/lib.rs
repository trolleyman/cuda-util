
pub use cuda_macros_impl::{host, device, global};

use ::cuda::ffi::driver_types::cudaStream_t;

#[repr(C)]
pub struct ExecutionConfig {
	pub grid_size: [usize; 3],
	pub block_size: [usize; 3],
	pub shared_mem_size: usize,
	pub cuda_stream: cudaStream_t,
}

macro_rules! usize_arr_type {
	(i1) => (usize);
	(i2) => ([usize; 2]);
	(i3) => ([usize; 3]);
}

macro_rules! usize_arr_expr {
	(i1, $name:ident) => ([$name, 1, 1]);
	(i2, $name:ident) => ([$name[0], $name[1], 1]);
	(i3, $name:ident) => ([$name[0], $name[1], $name[2]]);
}

macro_rules! impl_from_execution_config {
	($gs:ident, $bs:ident) => {
		impl From<(usize_arr_type!($gs), usize_arr_type!($bs), usize, ::cuda::ffi::driver_types::cudaStream_t)> for ExecutionConfig {
			fn from((grid_size, block_size, shared_mem_size, cuda_stream): (usize_arr_type!($gs), usize_arr_type!($bs), usize, ::cuda::ffi::driver_types::cudaStream_t)) -> ExecutionConfig {
				ExecutionConfig {
					grid_size: usize_arr_expr!($gs, grid_size),
					block_size: usize_arr_expr!($bs, block_size),
					shared_mem_size,
					cuda_stream,
				}
			}
		}
		impl From<(usize_arr_type!($gs), usize_arr_type!($bs), usize)> for ExecutionConfig {
			fn from((grid_size, block_size, shared_mem_size): (usize_arr_type!($gs), usize_arr_type!($bs), usize)) -> ExecutionConfig {
				ExecutionConfig {
					grid_size: usize_arr_expr!($gs, grid_size),
					block_size: usize_arr_expr!($bs, block_size),
					shared_mem_size,
					cuda_stream: ::std::ptr::null_mut() as ::cuda::ffi::driver_types::cudaStream_t,
				}
			}
		}
		impl From<(usize_arr_type!($gs), usize_arr_type!($bs))> for ExecutionConfig {
			fn from((grid_size, block_size): (usize_arr_type!($gs), usize_arr_type!($bs))) -> ExecutionConfig {
				ExecutionConfig {
					grid_size: usize_arr_expr!($gs, grid_size),
					block_size: usize_arr_expr!($bs, block_size),
					shared_mem_size: 0,
					cuda_stream: ::std::ptr::null_mut() as ::cuda::ffi::driver_types::cudaStream_t,
				}
			}
		}
	}
}

impl_from_execution_config!(i1, i1);
impl_from_execution_config!(i1, i2);
impl_from_execution_config!(i1, i3);

impl_from_execution_config!(i2, i1);
impl_from_execution_config!(i2, i2);
impl_from_execution_config!(i2, i3);

impl_from_execution_config!(i3, i1);
impl_from_execution_config!(i3, i2);
impl_from_execution_config!(i3, i3);
