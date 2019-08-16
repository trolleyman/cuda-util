
extern crate cuda_macros_build;

extern crate rustc_version;

pub fn main() {
	let mut opt = cuda_macros_build::BuildOptions::default();
	
	// Detect if nightly compiler, and if so, enable unstable mode
	if rustc_version::version_meta().map(|v| v.channel == rustc_version::Channel::Nightly).unwrap_or(false) {
		println!("cargo:rustc-cfg=unstable");
		opt.feature("unstable");
	}

	opt.build();
}
