
use cuda::ffi::driver_types::cudaStream_t;


#[repr(C)]
pub struct ExecutionConfig {
	pub grid_size: [u32; 3],
	pub block_size: [u32; 3],
	pub shared_mem_size: u32,
	pub cuda_stream: cudaStream_t,
}

macro_rules! u32_arr_type {
	(a1) => (u32);
	(a2) => ([u32; 2]);
	(a3) => ([u32; 3]);
	(t1) => ((u32,));
	(t2) => ((u32, u32));
	(t3) => ((u32, u32, u32));
}

macro_rules! u32_arr_expr {
	(a1, $name:ident) => ([$name, 1, 1]);
	(a2, $name:ident) => ([$name[0], $name[1], 1]);
	(a3, $name:ident) => ([$name[0], $name[1], $name[2]]);
	(t1, $name:ident) => ([$name.0, 1, 1]);
	(t2, $name:ident) => ([$name.0, $name.1, 1]);
	(t3, $name:ident) => ([$name.0, $name.1, $name.2]);
}

macro_rules! impl_from_execution_config {
	($gs:ident, $bs:ident) => {
		impl From<(u32_arr_type!($gs), u32_arr_type!($bs), u32, ::cuda::ffi::driver_types::cudaStream_t)> for ExecutionConfig {
			fn from((grid_size, block_size, shared_mem_size, cuda_stream): (u32_arr_type!($gs), u32_arr_type!($bs), u32, ::cuda::ffi::driver_types::cudaStream_t)) -> ExecutionConfig {
				ExecutionConfig {
					grid_size: u32_arr_expr!($gs, grid_size),
					block_size: u32_arr_expr!($bs, block_size),
					shared_mem_size,
					cuda_stream,
				}
			}
		}
		impl From<(u32_arr_type!($gs), u32_arr_type!($bs), u32)> for ExecutionConfig {
			fn from((grid_size, block_size, shared_mem_size): (u32_arr_type!($gs), u32_arr_type!($bs), u32)) -> ExecutionConfig {
				ExecutionConfig {
					grid_size: u32_arr_expr!($gs, grid_size),
					block_size: u32_arr_expr!($bs, block_size),
					shared_mem_size,
					cuda_stream: std::ptr::null_mut() as cudaStream_t,
				}
			}
		}
		impl From<(u32_arr_type!($gs), u32_arr_type!($bs))> for ExecutionConfig {
			fn from((grid_size, block_size): (u32_arr_type!($gs), u32_arr_type!($bs))) -> ExecutionConfig {
				ExecutionConfig {
					grid_size: u32_arr_expr!($gs, grid_size),
					block_size: u32_arr_expr!($bs, block_size),
					shared_mem_size: 0,
					cuda_stream: std::ptr::null_mut() as cudaStream_t,
				}
			}
		}
	}
}

impl_from_execution_config!(a1, a1);
impl_from_execution_config!(a1, a2);
impl_from_execution_config!(a1, a3);

impl_from_execution_config!(a2, a1);
impl_from_execution_config!(a2, a2);
impl_from_execution_config!(a2, a3);

impl_from_execution_config!(a3, a1);
impl_from_execution_config!(a3, a2);
impl_from_execution_config!(a3, a3);

impl_from_execution_config!(t1, t1);
impl_from_execution_config!(t1, t2);
impl_from_execution_config!(t1, t3);

impl_from_execution_config!(t2, t1);
impl_from_execution_config!(t2, t2);
impl_from_execution_config!(t2, t3);

impl_from_execution_config!(t3, t1);
impl_from_execution_config!(t3, t2);
impl_from_execution_config!(t3, t3);