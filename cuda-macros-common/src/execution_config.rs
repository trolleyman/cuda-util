
use cuda::ffi::driver_types::cudaStream_t;


/// CUDA execution configuration information.
/// 
/// Defines how to run a kernel, i.e. how many threads and blocks, how large shared memory is, and which stream to run the kernel on.
/// 
/// There are constructors defined from a tuple of the format `(grid_size, block_size[, shared_mem_size[, cuda_stream]])`.
/// `shared_mem_size` defaults to `0`. `cuda_stream` defaults to `NULL`, i.e. the default stream. `grid_size` and `block_size` are given as either
/// a single integer or a list of up to 3 integers (either as a tuple or fixed-length array). These three values define the sizes of the grid or block
/// in the three dimensions, `x`, `y`, and `z`. Missing dimensions are taken to equal `1`.
/// 
/// # Examples
/// ```
/// # use cuda::runtime::CudaStream;
/// # use cuda_macros_common::ExecutionConfig;
/// ExecutionConfig::from((1, 1));              // 1 grid, 1 block, 0 bytes of shared memory, default stream
/// ExecutionConfig::from((1, 1, 16));          // 1 grid, 1 block, 16 bytes of shared memory, default stream
/// ExecutionConfig::from((1, 1, 16, CudaStream::create().unwrap()));
///                                             // 1 grid, 1 block, 16 bytes of shared memory, custom stream
/// ExecutionConfig::from((16, 1));             // 16 grids, 1 block
/// ExecutionConfig::from((16, 16));            // 16 grids, 16 blocks, 16*16 = 256 threads total
/// ExecutionConfig::from(((4, 4), 1));         // 4x4 grids, 1 block, 4*4 = 16 threads total
/// ExecutionConfig::from((1, (4, 4)));         // 1 grid, 4x4 blocks, 1*4*4 = 16 threads total
/// ExecutionConfig::from(((4, 4), (4, 4)));    // 4x4 grids, 4x4 blocks, 4*4*4*4 = 16*16 = 256 threads total
/// ExecutionConfig::from(([2, 3, 4], [2, 2])); // You can also use fixed-length arrays
///                                             // 2x3x4 grids, 2x2 blocks, 2*3*4 * 2*2 = 24*4 = 96 threads total
/// ```
#[repr(C)]
pub struct ExecutionConfig {
	pub grid_size: [u32; 3],
	pub block_size: [u32; 3],
	pub shared_mem_size: u32,
	pub cuda_stream: cudaStream_t,
}

macro_rules! u32_arr_type {
	(a0) => (u32);
	(a1) => ([u32; 1]);
	(a2) => ([u32; 2]);
	(a3) => ([u32; 3]);
	(t1) => ((u32,));
	(t2) => ((u32, u32));
	(t3) => ((u32, u32, u32));
}

macro_rules! u32_arr_expr {
	(a0, $name:ident) => ([$name, 1, 1]);
	(a1, $name:ident) => ([$name[0], 1, 1]);
	(a2, $name:ident) => ([$name[0], $name[1], 1]);
	(a3, $name:ident) => ([$name[0], $name[1], $name[2]]);
	(t1, $name:ident) => ([$name.0, 1, 1]);
	(t2, $name:ident) => ([$name.0, $name.1, 1]);
	(t3, $name:ident) => ([$name.0, $name.1, $name.2]);
}

macro_rules! impl_from_execution_config {
	($gs:ident, $bs:ident) => {
		impl From<(u32_arr_type!($gs), u32_arr_type!($bs), u32, ::cuda::runtime::CudaStream)> for ExecutionConfig {
			fn from((grid_size, block_size, shared_mem_size, cuda_stream): (u32_arr_type!($gs), u32_arr_type!($bs), u32, ::cuda::runtime::CudaStream)) -> ExecutionConfig {
				ExecutionConfig {
					grid_size: u32_arr_expr!($gs, grid_size),
					block_size: u32_arr_expr!($bs, block_size),
					shared_mem_size,
					cuda_stream: cuda_stream.as_raw(),
				}
			}
		}
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

impl_from_execution_config!(a0, a0);
impl_from_execution_config!(a0, a1);
impl_from_execution_config!(a0, a2);
impl_from_execution_config!(a0, a3);

impl_from_execution_config!(a1, a0);
impl_from_execution_config!(a1, a1);
impl_from_execution_config!(a1, a2);
impl_from_execution_config!(a1, a3);

impl_from_execution_config!(a2, a0);
impl_from_execution_config!(a2, a1);
impl_from_execution_config!(a2, a2);
impl_from_execution_config!(a2, a3);

impl_from_execution_config!(a3, a0);
impl_from_execution_config!(a3, a1);
impl_from_execution_config!(a3, a2);
impl_from_execution_config!(a3, a3);

impl_from_execution_config!(t1, a0);
impl_from_execution_config!(t1, a1);
impl_from_execution_config!(t1, a2);
impl_from_execution_config!(t1, a3);

impl_from_execution_config!(t2, a0);
impl_from_execution_config!(t2, a1);
impl_from_execution_config!(t2, a2);
impl_from_execution_config!(t2, a3);

impl_from_execution_config!(t3, a0);
impl_from_execution_config!(t3, a1);
impl_from_execution_config!(t3, a2);
impl_from_execution_config!(t3, a3);

impl_from_execution_config!(a0, t1);
impl_from_execution_config!(a0, t2);
impl_from_execution_config!(a0, t3);

impl_from_execution_config!(a1, t1);
impl_from_execution_config!(a1, t2);
impl_from_execution_config!(a1, t3);

impl_from_execution_config!(a2, t1);
impl_from_execution_config!(a2, t2);
impl_from_execution_config!(a2, t3);

impl_from_execution_config!(a3, t1);
impl_from_execution_config!(a3, t2);
impl_from_execution_config!(a3, t3);

impl_from_execution_config!(t1, t1);
impl_from_execution_config!(t1, t2);
impl_from_execution_config!(t1, t3);

impl_from_execution_config!(t2, t1);
impl_from_execution_config!(t2, t2);
impl_from_execution_config!(t2, t3);

impl_from_execution_config!(t3, t1);
impl_from_execution_config!(t3, t2);
impl_from_execution_config!(t3, t3);
