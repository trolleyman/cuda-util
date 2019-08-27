
use std::fs;
use std::io;
use std::path::PathBuf;

use lazy_static::lazy_static;
use proc_macro2::TokenStream;
use fs2::FileExt;

use util::FnInfo;
use util::write::TransError;


lazy_static!{
	pub(crate) static ref CUDA_MACROS_OUT_DIR: Option<PathBuf> = std::env::var_os("CUDA_MACROS_OUT_DIR").map(|p| p.into());
}


pub fn output_fn(f: &syn::ItemFn, fn_info: &FnInfo) -> Result<(), TokenStream> {
	match output_fn_impl(f, fn_info) {
		Ok(()) => Ok(()),
		Err(TransError::IoError(e)) => Err(syn::Error::new_spanned(f.clone(), format!("failed to write function: {:?}", e)).to_compile_error()),
		Err(TransError::SynError(e)) => Err(e.to_compile_error())
	}
}

fn output_fn_impl(f: &syn::ItemFn, fn_info: &FnInfo) -> Result<(), TransError> {
	if let Some(dir) = CUDA_MACROS_OUT_DIR.clone() {
		// Output to dir
		let lock_path = dir.join(".lock");
		let header_path = dir.join("header.h");
		
		// Get lock for header file
		let lock = fs::OpenOptions::new().read(true).write(true).create(true).open(lock_path)?;
		lock.lock_exclusive()?;
		
		// Open/create header file
		let mut header_file = fs::OpenOptions::new().read(true).append(true).create(true).open(header_path)?;
		// Open src file
		let src_path = dir.join("source.cu");
		let mut src_file = fs::OpenOptions::new().read(true).write(true).create(true).open(src_path)?;
		
		// Write function to files
		util::write::write_fn(&mut header_file, &mut src_file, f, fn_info)?;
		
		header_file.sync_all()?;
		src_file.sync_all()?;

		lock.unlock()?;
		drop(lock);
		Ok(())
	} else {
		// Null output
		let mut header_file = io::Cursor::new(Vec::new());
		let mut src_file = io::Cursor::new(Vec::new());
		util::write::write_fn(&mut header_file, &mut src_file, f, fn_info)?;
		Ok(())
	}
}
