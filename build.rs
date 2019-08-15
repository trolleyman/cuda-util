
extern crate cuda_macros_build;

extern crate rustc_version;

pub fn main() {
	let mut opt = cuda_macros_build::BuildOptions::default();
	
	// Detect if nightly compiler, and if so, enable specialization
	if rustc_version::version_meta().map(|v| v.channel == rustc_version::Channel::Nightly).unwrap_or(false) {
		println!("cargo:rustc-cfg=specialization");
		opt.feature("specialization");
	}

	opt.build();
}
