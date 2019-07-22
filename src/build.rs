
extern crate tempdir;

use std::path::PathBuf;
use std::process::Command;
use std::fs;

use tempdir::TempDir;


pub fn build() {
	let out_dir = PathBuf::from(std::env::var_os("OUT_DIR")
		.expect("expected to be called in a build.rs script")).join("cuda_macros");
	let dir = out_dir.join("src");
	let actual_target_dir = out_dir.join("target");
	let temp = TempDir::new("cuda-macros").unwrap();
	let target_dir = temp.path().join("target");

	// Ensure output directory is empty & exists
	if dir.is_file() {
		fs::remove_file(&dir).unwrap();
	}
	if dir.is_dir() {
		// Remove old dir
		fs::remove_dir_all(&dir).unwrap();
	}
	fs::create_dir_all(&dir).unwrap();

	// Compile crate & output sources
	let mut command = Command::new(env!("CARGO"));
	command.arg("check")
		.arg("--all-targets")
		.arg("--target")
		.arg(std::env::var("TARGET").unwrap())
		.arg("--target-dir")
		.arg(target_dir)
		.env("CUDA_MACROS_OUT_DIR", dir);

	let mut features = vec![];
	for (key, _) in std::env::vars() {
		if key.starts_with("CARGO_FEATURE_") {
			features.push(key["CARGO_FEATURE_".len()..].to_string())
		}
	}
	command.arg("--features").arg(features.join(" "));

	if std::env::var("PROFILE").unwrap() == "release" {
		command.arg("--release");
	}
	let status = command.status()
		.expect("failed to execute cargo check");

	if status.success() {
		panic!("failed to execute cargo check");
	}

	// Compile sources that were output at the previous step
	let libname = format!("cuda_macros_{}_{}", std::env::var("CARGO_PKG_NAME").unwrap(), std::env::var("CARGO_PKG_VERSION").unwrap());
	// TODO
	cc::Build::new().cuda(true).compile(&libname);
	unimplemented!()
}
