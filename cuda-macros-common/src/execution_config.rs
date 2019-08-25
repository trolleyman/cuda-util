
use cuda::runtime::CudaStream;
use cuda::ffi::driver_types::cudaStream_t;

/// Shorthand for constructing [`Dim3`](struct.Dim3.html)
#[macro_export]
macro_rules! dim3 {
	() => ( $crate::Dim3::default() );
	($x:expr) => ( $crate::Dim3::new1($x) );
	($x:expr, $y:expr) => ( $crate::Dim3::new2($x, $y) );
	($x:expr, $y:expr, $z:expr) => ( $crate::Dim3::new3($x, $y, $z) );
}
/// 3-dimensional size specification.
/// 
/// See [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dim3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dim3)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Dim3 {
	/// `x`-dimension size
	pub x: u32,
	/// `y`-dimension size
	pub y: u32,
	/// `z`-dimension size
	pub z: u32,
}
impl Dim3 {
	/// Constructs a new `Dim3` with the specified values.
	pub fn new(x: u32, y: u32, z: u32) -> Self {
		Dim3 {
			x, y, z
		}
	}

	/// Equivalent to `Dim3::new(x, 1, 1)`.
	pub fn new1(x: u32) -> Self {
		Dim3::new(x, 1, 1)
	}

	/// Equivalent to `Dim3::new(x, y, 1)`.
	pub fn new2(x: u32, y: u32) -> Self {
		Dim3::new(x, y, 1)
	}

	/// Equivalent to `Dim3::new(x, y, z)`.
	pub fn new3(x: u32, y: u32, z: u32) -> Self {
		Dim3::new(x, y, z)
	}
}
impl Default for Dim3 {
	fn default() -> Self {
		Dim3::new(1, 1, 1)
	}
}
impl From<u32> for Dim3 {
	fn from(x: u32) -> Dim3 {
		Dim3::new1(x.into())
	}
}
impl From<(u32,)> for Dim3 {
	fn from((x,): (u32,)) -> Dim3 {
		Dim3::new1(x.into())
	}
}
impl From<(u32, u32)> for Dim3 {
	fn from((x, y): (u32, u32)) -> Dim3 {
		Dim3::new2(x.into(), y.into())
	}
}
impl From<(u32, u32, u32)> for Dim3 {
	fn from((x, y, z): (u32, u32, u32)) -> Dim3 {
		Dim3::new3(x.into(), y.into(), z.into())
	}
}
impl From<[u32; 1]> for Dim3 {
	fn from(x: [u32; 1]) -> Dim3 {
		Dim3::new1(x[0].into())
	}
}
impl From<[u32; 2]> for Dim3 {
	fn from(x: [u32; 2]) -> Dim3 {
		Dim3::new2(x[0].into(), x[1].into())
	}
}
impl From<[u32; 3]> for Dim3 {
	fn from(x: [u32; 3]) -> Dim3 {
		Dim3::new3(x[0].into(), x[1].into(), x[2].into())
	}
}

fn div_ceil(x: usize, divisor: usize) -> usize {
	x / divisor + (x % divisor != 0) as usize
}

/// CUDA execution configuration information.
/// 
/// Defines how to run a kernel, i.e. how many threads and blocks, how large shared memory is, and which stream to run the kernel on.
/// 
/// There are constructors defined from a tuple of the format `(Dg, Db[, Ns[, S]])`, where:
/// - `Dg` specifies the dimension and size of the grid, such that `Dg.x * Dg.y * Dg.z` equals the number of blocks being launched.
/// - `Db` specifies the dimension and size of each block, such that `Db.x * Db.y * Db.z` equals the number of threads per block.
/// - `Ns` specifies the number of bytes in shared memory that is dynamically allocated *per block* for this call. This can be accessed by using a `#[shared]`
///   attribute on a `let <ident>: [<type>];` expression. This defaults to 0.
/// - `S` specifies the associated CUDA stream. This defaults to `NULL`.
/// 
/// `Dg` and `Db` are 3-dimensional values, but dimensions can be ommitted, and their size will default to 1. 1-, 2- and 3- tuples are accepted.
/// 
/// # Examples
/// ```
/// # use cuda::runtime::CudaStream;
/// # use cuda_macros_common::ExecutionConfig;
/// ExecutionConfig::from((1, 1));              // 1x1x1 grid of 1x1x1 blocks, 0 bytes of shared memory, default stream
/// ExecutionConfig::from((1, 1, 16usize));     // 1x1x1 grid of 1x1x1 blocks, 16 bytes of shared memory, default stream
/// ExecutionConfig::from((1, 1, 16usize, CudaStream::create().unwrap()));
///                                             // 1x1x1 grid of 1x1x1 block, 16 bytes of shared memory, custom stream
/// ExecutionConfig::from((1, 16));             // 1x1x1 grid of 16x1x1 blocks, 16 threads total
/// ExecutionConfig::from((16, 16));            // 16x1x1 grid of 16x1x1 blocks, 16*16 = 256 threads total
/// ExecutionConfig::from((1, (4, 4)));         // 1x1x1 grid of 4x4x1 blocks, 4*4 = 16 threads total
/// ExecutionConfig::from(((4, 4), 1));         // 4x4x1 grid of 1x1x1 blocks, 4*4 = 16 threads total
/// ExecutionConfig::from(((4, 4), (4, 4)));    // 4x4x1 grid of 4x4x1 blocks, 4*4 * 4*4 = 16*16 = 256 threads total
/// ExecutionConfig::from(([2, 2], [2, 3, 4])); // You can also use fixed-length arrays
///                                             // 2x2x1 grid of 2x3x4 blocks, 2*3*4 * 2*2 = 24*4 = 96 threads total
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct ExecutionConfig {
	/// Dimension and size of the grid, i.e. `Dg`
	pub grid_size: Dim3,
	/// Dimension and size of each block, i.e. `Db`
	pub block_size: Dim3,
	/// Number of bytes in shared memory, i.e. `Ns`
	pub shared_mem_size: usize,
	/// CUDA associated stream, i.e. `S`
	pub cuda_stream: cudaStream_t,
}
impl ExecutionConfig {
	/// Number of threads per block, by default. Set to `1024`.
	pub const DEFAULT_NUM_THREADS_PER_BLOCK: u32 = 1024;

	/// Returns an `ExecutionConfig` with `num_threads` **or more** threads total, arranged in
	/// blocks of dimension `1024x1x1` or less.
	pub fn from_num_threads(num_threads: u32) -> Self {
		ExecutionConfig::from_num_threads_with_shared_mem_per_thread(num_threads, 0)
	}

	/// Returns an `ExecutionConfig` with `num_threads` **or more** threads total, arranged in
	/// blocks of dimension `1024x1x1` or less, and with `shared_mem_size` bytes of shared storage
	/// per thread.
	pub fn from_num_threads_with_shared_mem_per_thread(num_threads: u32, shared_mem_size: usize) -> Self {
		let num_threads_per_block = if num_threads < ExecutionConfig::DEFAULT_NUM_THREADS_PER_BLOCK {
			num_threads
		} else {
			ExecutionConfig::DEFAULT_NUM_THREADS_PER_BLOCK
		};
		let num_blocks = div_ceil(num_threads as usize, num_threads_per_block as usize) as u32;

		ExecutionConfig::from((num_blocks, num_threads_per_block, num_threads_per_block as usize * shared_mem_size))
	}
}

impl<A, B> From<(A, B)> for ExecutionConfig where A: Into<Dim3>, B: Into<Dim3> {
	#[inline]
	fn from((grid_size, block_size): (A, B)) -> ExecutionConfig {
		ExecutionConfig::from((grid_size.into(), block_size.into(), 0usize))
	}
}
impl<A, B, C> From<(A, B, C)> for ExecutionConfig where A: Into<Dim3>, B: Into<Dim3>, C: Into<usize> {
	#[inline]
	fn from((grid_size, block_size, shared_mem_size): (A, B, C)) -> ExecutionConfig {
		ExecutionConfig::from((grid_size, block_size, shared_mem_size, CudaStream::default()))
	}
}
impl<A, B, C> From<(A, B, C, CudaStream)> for ExecutionConfig where A: Into<Dim3>, B: Into<Dim3>, C: Into<usize> {
	#[inline]
	fn from((grid_size, block_size, shared_mem_size, cuda_stream): (A, B, C, CudaStream)) -> ExecutionConfig {
		ExecutionConfig::from((grid_size, block_size, shared_mem_size, cuda_stream.as_raw()))
	}
}
impl<A, B, C> From<(A, B, C, cudaStream_t)> for ExecutionConfig where A: Into<Dim3>, B: Into<Dim3>, C: Into<usize> {
	#[inline]
	fn from((grid_size, block_size, shared_mem_size, cuda_stream): (A, B, C, cudaStream_t)) -> ExecutionConfig {
		ExecutionConfig {
			grid_size: grid_size.into(),
			block_size: block_size.into(),
			shared_mem_size: shared_mem_size.into(),
			cuda_stream
		}
	}
}
