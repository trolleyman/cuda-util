
use std::collections::hash_map::DefaultHasher;
use std::fs::{self, File};
use std::path::PathBuf;
use std::io::{self, Seek, SeekFrom};

use fs2::FileExt;

use cuda_macros_util as util;
use cuda_macros_util::FunctionType;


pub fn output_fn(f: &syn::ItemFn, fn_type: FunctionType) -> Result<(), TokenStream> {
	let dir = PathBuf::from(env!("CUDA_MACROS_OUT_DIR"));
	output_fn_impl(dir, f, fn_type)
		.map_err(|e| syn::Error::new(f.clone(), format!("failed to write function to ouput dir `{}`: {}", dir.display(), e)))
}

fn output_fn_impl(dir: PathBuf, f: &syn::ItemFn, fn_type: FunctionType) -> io::Result<()> {
	let mut hasher = DefaultHasher::new();
	f.hash(&mut hasher);
	let hash = hasher.finish();
	
	let header_path = dir.join("rust_cuda_macros_header.h");

	// Open/create header file
	let header_file = fs::OpenOptions::new().read(true).append(true).create(true).open()?.lock_exclusive();
	util::write::write_fn_header_file(header_file, f, fn_type);
	header_file.sync_all()?;
	drop(header_file);

	let src_path = dir.join(format!("{}_{:0x}.cu", f.ident, hash));

	let mut src_file = File::create(src_path)?;
	util::write::write_fn_src_file(src_file, f, fn_type)?;
	src_file.sync_all()?;
	Ok(())
}
