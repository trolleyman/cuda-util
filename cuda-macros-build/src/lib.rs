#![doc(html_logo_url = "https://raw.githubusercontent.com/trolleyman/cuda-util/master/cuda-macros-build/res/logo_512.png")]
//! Outputs & compiles the CUDA-annotated functions in the current crate.
//! 
//! The [`build()`](fn.build.html) function is where the magic happens.
//! 
//! It essentially calls `cargo check` on the current crate with an environment variable
//! (`CUDA_MACROS_OUT_DIR`) set to the current output dir.
//! 
//! The macros in [`cuda-macros-impl`](https://docs.rs/cuda-macros-impl) are then called
//! and output CUDA C code to this directory.
//! 
//! Control then returns to this build script which compiles the CUDA code using `nvcc`,
//! and instructs Cargo to link to it when building the crate.

extern crate cc;
extern crate whoami;
extern crate serde_json;


use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::fs;
use std::io::{self, prelude::*};
use std::time::{SystemTime, Duration};

use lazy_static::lazy_static;

use serde_json::Value;


lazy_static! {
	static ref BUILD_DIRNAME: String = format!("cuda-macros_build_{}_{}", std::env::var("CARGO_PKG_NAME").unwrap(), std::env::var("CARGO_PKG_VERSION").unwrap());
}

/// Set of build options to apply when building this crate.
pub struct BuildOptions {
	pub(crate) features: Vec<String>,
}
impl BuildOptions {
	/// Constructs the default build options.
	pub fn new() -> Self {
		BuildOptions {
			features: Vec::new(),
		}
	}

	/// Adds a feature to enable with the specified name.
	pub fn feature(&mut self, feature: impl Into<String>) -> &mut Self {
		self.features.push(feature.into());
		self
	}

	/// Performs the build with the specified options.
	pub fn build(self) {
		build_opt(self);
	}
}
impl Default for BuildOptions {
	fn default() -> Self {
		BuildOptions::new()
	}
}


fn update_modified_file(target_dir: impl AsRef<Path>) -> io::Result<()> {
	let mut modified_file = fs::File::create(target_dir.as_ref().join("invoked.timestamp"))?;
	writeln!(modified_file, "This file has an mtime of when this was last built.")?;
	modified_file.sync_all()?;
	Ok(())
}

fn should_cleanup_dir(p: impl AsRef<Path>) -> io::Result<bool> {
	let p = p.as_ref();
	let name = p.file_name();
	if name.is_none() {
		return Ok(false);
	}
	let name = name.map(|p| p.to_string_lossy().to_string()).unwrap_or("".to_string());
	if !name.starts_with("cuda-macros_build_") || name == BUILD_DIRNAME.as_str() {
		return Ok(false);
	}

	// Check modified time
	let modified = fs::metadata(p.join("invoked.timestamp"))?.modified()?;
	let now = SystemTime::now();
	// One month
	let max_duration = Duration::from_secs(60 * 60 * 24 * 30);
	Ok(now.duration_since(modified).unwrap_or(Duration::new(0, 0)) > max_duration)
}

/// Loop through folders in temp dir & cleanup cuda-macros builds that have not been modified in a long time
fn cleanup_build_dirs() -> io::Result<()> {
	for entry in fs::read_dir(std::env::temp_dir())? {
		let entry = entry?;
		let ty = entry.file_type()?;
		if should_cleanup_dir(entry.path()).unwrap_or(true) {
			if ty.is_dir() {
				fs::remove_dir_all(entry.path())?;
			} else {
				fs::remove_file(entry.path())?;
			}
		}
	}
	Ok(())
}

fn is_cargo_cmd_valid() -> bool {
	match Command::new(env!("CARGO")).arg("-V").output() {
		Ok(out) => out.stdout.starts_with(b"cargo "),
		Err(_) => false
	}
}

/// Performs the build of the current crate with the default [`BuildOptions`](struct.BuildOptions.html).
pub fn build() {
	build_opt(BuildOptions::default());
}

fn build_opt(opt: BuildOptions) {
	if std::env::var_os("CUDA_MACROS_BUILD_SCRIPT").is_some() {
		println!("cargo:rerun-if-env-changed=CUDA_MACROS_BUILD_TIMESTAMP");
		// Detected recursion: exit
		return;
	}
	std::env::set_var("CUDA_MACROS_BUILD_SCRIPT", "1");

	cleanup_build_dirs().ok();

	let crate_out_dir = PathBuf::from(std::env::var_os("OUT_DIR")
		.expect("expected to be called in a build.rs script"));
	let out_dir = crate_out_dir.join("cuda-macros").join("src");

	// Create target dir that is deterministic on the package name & version being built
	// This allows incremental build, without clashing with the current cargo.exe that it is building
	let target_dir = crate_out_dir
		.join("..").join("..").join("..").join("..")
		.canonicalize().unwrap()
		.join(&*BUILD_DIRNAME);
	if !target_dir.is_dir() {
		fs::create_dir_all(&target_dir).unwrap();
	}

	// Create/update .modified file
	update_modified_file(&target_dir).ok();

	// Ensure output directory is empty & exists
	if out_dir.is_file() {
		fs::remove_file(&out_dir).unwrap();
	}
	if out_dir.is_dir() {
		// Remove old dir
		fs::remove_dir_all(&out_dir).unwrap();
	}
	fs::create_dir_all(&out_dir).unwrap();

	// Check if we are cargo & not rls
	if !is_cargo_cmd_valid() {
		return;
	}

	// Get metadata
	let metadata = Command::new(env!("CARGO"))
		.arg("metadata")
		.arg("--format-version")
		.arg("1")
		.arg("--no-deps")
		.output()
		.expect("cargo metadata failed to run");
	if !metadata.status.success() {
		// Ignore - the actual compilation of this crate will catch any errors, and show them better than we can display them
		return;
	}
	let metadata: Value = serde_json::from_slice(&metadata.stdout).unwrap();

	let mut valid_features = HashSet::new();
	for package in metadata["packages"].as_array().unwrap().iter() {
		for feature in package["features"].as_object().unwrap().keys() {
			if feature != "default" {
				valid_features.insert(feature.clone());
			}
		}
	}

	// Compile crate & output sources
	let unix_now = SystemTime::UNIX_EPOCH.elapsed().unwrap_or_default();
	let mut command = Command::new(env!("CARGO"));
	command.arg("check")
		//.arg("--all-targets") // TODO: Figure out which targets to compile (--lib, --bins, --tests, --benches, --examples)
		.arg("--target")
		.arg(std::env::var("TARGET").unwrap())
		.arg("--target-dir")
		.arg(target_dir)
		.arg("--color=always")
		.arg("--profile=test")  // check #[cfg(test)] parts of the program as well
		.env("CUDA_MACROS_OUT_DIR", &out_dir)
		.env("CUDA_MACROS_BUILD_TIMESTAMP", format!("{}.{}", unix_now.as_secs(), unix_now.subsec_nanos()));

	let mut features = vec![];
	for (key, _) in std::env::vars() {
		if key.starts_with("CARGO_FEATURE_") {
			let feature = &key["CARGO_FEATURE_".len()..];
			let feature: String = feature.chars().flat_map(|c| c.to_lowercase()).collect();
			
			// Check if feature is valid
			if valid_features.contains(&feature) {
				features.push(feature)
			}
		}
	}
	for f in &opt.features {
		features.push(f.clone());
	}
	if features.len() > 0 {
		command.arg("--features");
		command.arg(features.join(" "));
	}

	if std::env::var("PROFILE").unwrap() == "release" {
		command.arg("--release");
	}
	let output = command.output()
		.expect("failed to execute cargo check");

	if !output.status.success() {
		// TODO: Ignore -- this should be fine, but might not due to bugs in implementation, as all errors that occur here
		// should be picked up anyway in the actual pass that happens after the build phase.
		//println!("cargo check failed");
		io::stdout().lock().write_all(&output.stdout).ok();
		io::stderr().lock().write_all(&output.stderr).ok();
		std::env::set_var("RUST_BACKTRACE", "0");
		panic!("compilation error");
	}

	// TODO: Test compile a CUDA file before main

	// Compile sources that were output at the previous step
	// TODO: Hanlde -gencode arg
	let libname = format!("cuda_macros_{}_{}", std::env::var("CARGO_PKG_NAME").unwrap(), std::env::var("CARGO_PKG_VERSION").unwrap());
	cc::Build::new()
		.cuda(true)
		.include(&out_dir)
		.file(&out_dir.join("source.cu"))
		.compile(&libname);
}
