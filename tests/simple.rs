
use cuda_util::*;

#[test]
fn simple_cpu() {
	let mut t: CpuTensor<f32, _> = CpuTensor::eye(3);
	t *= 0;
	assert_eq!(t, CpuTensor::zeros((3, 3)));
}

#[test]
fn simple_gpu() {
	let mut t: GpuTensor<f32, _> = GpuTensor::eye(3);
	t *= 0;
	assert_eq!(t, GpuTensor::zeros((3, 3)));
}
