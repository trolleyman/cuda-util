
pub use cuda_macros_impl::{host, device, global};

pub struct ExecutionConfig {
    pub grid_size: [usize; 3];
    pub block_size: [usize; 3];
    pub shared_mem_size: usize;
    pub cuda_stream: usize;
}
impl From<(usize, usize)> for ExecutionConfig {
    fn from((grid_size, block_size): (usize, usize)) -> ExecutionConfig {
        ExecutionConfig {
            grid_size: [grid_size, 1, 1],
            block_size: [block_size, 1, 1],
            shared_mem_size: 0,
            cuda_stream: 0,
        }
    }
}
