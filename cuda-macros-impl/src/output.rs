
use std::collections::hash_map::DefaultHasher;
use std::fs::{self, File};
use std::io;
use std::path::PathBuf;
use std::hash::{Hash, Hasher};

use lazy_static::lazy_static;
use proc_macro2::TokenStream;
use fs2::FileExt;

use cuda_macros_util as util;
use util::FunctionType;
use util::write::TransError;


lazy_static!{
	pub(crate) static ref CUDA_MACROS_OUT_DIR: Option<PathBuf> = std::env::var_os("CUDA_MACROS_OUT_DIR").map(|p| p.into());
}


pub fn output_fn(f: &syn::ItemFn, fn_type: FunctionType) -> Result<(), TokenStream> {
	match output_fn_impl(f, fn_type) {
		Ok(()) => Ok(()),
		Err(TransError::IoError(e)) => Err(syn::Error::new_spanned(f.clone(), format!("failed to write function: {}", e)).to_compile_error()),
		Err(TransError::SynError(e)) => Err(e.to_compile_error())
	}
}

fn output_fn_impl(f: &syn::ItemFn, fn_type: FunctionType) -> Result<(), TransError> {
	let mut hasher = DefaultHasher::new();
	f.hash(&mut hasher);
	let hash = hasher.finish();
	let hash = format!("{:0x}", hash);

	if let Some(dir) = CUDA_MACROS_OUT_DIR.clone() {
		// Output to dir
		let header_path = dir.join("rust_cuda_macros_header.h");
		
		// Open/create header file
		let mut header_file = fs::OpenOptions::new().read(true).append(true).create(true).open(header_path)?;
		header_file.lock_exclusive()?;
		
		// Write to header file
		util::write::write_fn_header_file(&mut header_file, f, fn_type, &hash)?;
		header_file.sync_all()?;
		drop(header_file);
		
		// Open src file
		let src_path = dir.join(format!("{}_{}.cu", f.ident, &hash));
		let mut src_file = File::create(src_path)?;

		// Write to src file
		util::write::write_fn_cu_file(&mut src_file, f, fn_type, &hash)?;
		src_file.sync_all()?;
		Ok(())
	} else {
		// Null output to header file
		let mut header_file = io::Cursor::new(Vec::new());
		util::write::write_fn_header_file(&mut header_file, f, fn_type, &hash)?;
		
		// Null output to src file
		let mut src_file = io::Cursor::new(Vec::new());
		util::write::write_fn_cu_file(&mut src_file, f, fn_type, &hash)?;
		Ok(())
	}
}
