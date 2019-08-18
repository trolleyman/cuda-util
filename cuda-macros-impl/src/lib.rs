
extern crate cuda_macros_common;

extern crate proc_macro;
extern crate cuda;
extern crate proc_macro2;
extern crate syn;
extern crate quote;
extern crate fs2;

mod output;

use proc_macro2::TokenStream;
use syn::{parse_macro_input, spanned::Spanned, punctuated::Punctuated};
use quote::{quote, ToTokens};
use quote::TokenStreamExt;

use cuda_macros_common::{conv, FunctionType, FnInfo, CudaNumberType};


fn process_host_fn(f: syn::ItemFn) -> Result<TokenStream, TokenStream> {
	// Pass through
	// TODO: handle cfg!(__CUDA_ARCH__)
	// https://stackoverflow.com/questions/23899190/can-a-host-device-function-know-where-it-is-executing
	// https://docs.nvidia.com/cuda/archive/9.0/cuda-c-programming-guide/index.html#host
	Ok(quote!{ #f })
}

fn process_device_fn(f: syn::ItemFn) -> Result<TokenStream, TokenStream> {
	// Output function
	let fn_info = FnInfo::new(&f, FunctionType::Device).map_err(|e| e.to_compile_error())?;
	output::output_fn(&f, &fn_info)?;

	/*
	let ident = f.ident;
	let vis = f.vis;

	// Return dummy identifier that will cause a compilation error when called
	// TODO: give a better compile time error somehow (maybe custom type?)
	quote!{
		#vis const #ident: () = ();
	}*/
	Ok(TokenStream::new())
}

fn instantiate_generic_type(ty: &syn::Type, tys: &[(&str, CudaNumberType)]) -> syn::Type {
	use syn::Type;

	let path = match cuda_macros_common::get_type_path(ty) {
		Ok(p) => p,
		// Ignore if type can't be converted into a type path
		Err(_) => return ty.clone(),
	};
	for (gen_ty_name, gen_ty) in tys {
		if path.is_ident(gen_ty_name) {
			// Create type from `gen_ty.rust_type()`, i.e. `u64`, `f32`, etc.
			let mut path_segments = Punctuated::new();
			path_segments.push_value(syn::PathSegment {
				ident: syn::Ident::new(gen_ty.rust_name(), ty.span()),
				arguments: syn::PathArguments::None
			});
			return Type::Path(syn::TypePath {
				qself: None,
				path: syn::Path {
					leading_colon: None,
					segments: path_segments
				}
			});
		}
	}
	ty.clone()
}

fn instantiate_generic_arg(arg: &syn::FnArg, tys: &[(&str, CudaNumberType)]) -> syn::FnArg {
	use syn::FnArg;

	match arg {
		FnArg::SelfRef(_) | FnArg::SelfValue(_) | FnArg::Inferred(_) => arg.clone(),
		FnArg::Captured(arg) => {
			FnArg::Captured(syn::ArgCaptured {
				pat: arg.pat.clone(),
				colon_token: arg.colon_token.clone(),
				ty: instantiate_generic_type(&arg.ty, tys)
			})
		},
		FnArg::Ignored(ty) => {
			FnArg::Ignored(instantiate_generic_type(ty, tys))
		}
	}
}

fn instantiate_generic_args(args: &Punctuated<syn::FnArg, syn::token::Comma>, tys: &[(&str, CudaNumberType)]) -> Punctuated<syn::FnArg, syn::token::Comma> {
	use syn::punctuated::Pair;

	args.pairs().map(|pair| match pair {
		Pair::Punctuated(arg, p) => {
			Pair::Punctuated(instantiate_generic_arg(arg, tys), p.clone())
		},
		Pair::End(arg) => {
			Pair::End(instantiate_generic_arg(arg, tys))
		}
	}).collect()
}

fn process_global_fn(f: syn::ItemFn) -> Result<TokenStream, TokenStream> {
	// Output function
	let fn_info = FnInfo::new(&f, FunctionType::Global).map_err(|e| e.to_compile_error())?;
	output::output_fn(&f, &fn_info)?;

	// Create function wrapper & link to CUDA function
	use syn::{FnArg, Pat};

	if f.unsafety.is_none() {
		return Err(syn::Error::new_spanned(f, "#[global] functions are required to be marked as unsafe").to_compile_error());
	}

	let args_decl = f.decl.inputs;
	let mut args = vec![];
	for arg in args_decl.iter() {
		match arg {
			FnArg::Captured(arg) => {
				let ident = match &arg.pat {
					Pat::Wild(_) => continue,
					Pat::Ident(item) => item.ident.clone(),
					_ => panic!("*Should* never happen due to `check_function_signature`")
				};
				args.push(ident);
			},
			FnArg::Ignored(_) => continue,
			_ => panic!("*Should* never happen due to `check_function_signature`")
		}
	}

	let args = quote!{ #(#args),* };

	let vis = f.vis;
	let ret = f.decl.output;
	let generics_params = f.decl.generics.params;

	let fn_ident = f.ident.clone();
	let fn_ident_c_base = format!("rust_cuda_macros_wrapper_{}_{}", fn_info.number_generics.len(), fn_ident);

	if !fn_info.is_generic() {
		let fn_ident_c = syn::Ident::new(&fn_ident_c_base, fn_ident.span());

		let extern_c_fn = quote!{ extern "C" { #vis fn #fn_ident_c(config: ::cuda_macros::ExecutionConfig, #args_decl) #ret; } };

		let wrapper_fn = quote!{
			#vis unsafe fn #fn_ident(config: impl ::std::convert::Into<::cuda_macros::ExecutionConfig>, #args_decl) #ret {
				let config = config.into();
				#fn_ident_c(config, #args)
			}
		};

		Ok(quote!{
			#extern_c_fn
			#wrapper_fn
		})
	} else {
		// Generics
		let mut tokens = TokenStream::new();
		let mut fn_ident_c = String::with_capacity(fn_ident_c_base.len());
		let mut tys_extra = vec![];
		cuda_macros_common::permutations(CudaNumberType::types(), fn_info.number_generics.len(), |tys| {
			// Get correct ident name (e.g. compare_f32_u32 is from compare<T: CudaNumber, U: CudaNumber>(x: T, y: T) with f32 and u32 type args)
			fn_ident_c.clone_from(&fn_ident_c_base);
			for ty in tys.iter() {
				fn_ident_c.push_str("_");
				fn_ident_c.push_str(ty.rust_name());
			}
			let fn_ident_c = syn::Ident::new(&fn_ident_c, fn_ident.span());

			// Instantiate generics in #args_decl
			tys_extra.clear();
			for (gen_ty_name, gen_ty) in fn_info.number_generics.iter().zip(tys.iter()) {
				tys_extra.push((gen_ty_name.as_str(), *gen_ty));
			}
			let args_decl_instantiated = instantiate_generic_args(&args_decl, &tys_extra[..]);
			tokens.extend(quote!{ extern "C" { #vis fn #fn_ident_c(config: ::cuda_macros::ExecutionConfig, #args_decl_instantiated) #ret; } });
		});

		// <T: CudaNumber, etc.>
		let mut generics = TokenStream::new();
		if let Some(lt_token) = &f.decl.generics.lt_token {
			generics.extend(quote!{ #lt_token });
		} else {
			generics.extend(quote!{ < });
		}
		generics.extend(quote!{ #generics_params });
		if let Some(gt_token) = &f.decl.generics.gt_token {
			generics.extend(quote!{ #gt_token });
		} else {
			generics.extend(quote!{ > });
		}
		// let generics = quote!{ <#generics_params> };
		
		// where T: CudaNumber
		let mut where_clause = TokenStream::new();
		if let Some(clause) = f.decl.generics.where_clause {
			where_clause.extend(quote!{ #clause });
		}

		let wrapper_fn = quote!{
			#vis unsafe fn #fn_ident#generics(config: impl ::std::convert::Into<::cuda_macros::ExecutionConfig>, #args_decl) #ret #where_clause {
				let config = config.into();
				// TODO
				#fn_ident_c(config, #args)
			}
		};
		panic!("{}", wrapper_fn);

		Ok(quote!{
			#tokens
			#wrapper_fn
		})
	}
}

fn process_fn(mut f: syn::ItemFn, fn_type: FunctionType) -> TokenStream {
	fn infallible_unwrap<T>(val: Result<T, T>) -> T {
		match val {
			Ok(t) => t,
			Err(t) => t,
		}
	}
	use cuda_macros_common::FunctionType::*;

	let mut device_host = None;
	for (i, a) in f.attrs.iter().enumerate() {
		if let Some(t) = FunctionType::try_from_attr(a) {
			match (fn_type, t) {
				(Host, Device) | (Device, Host) => {
					device_host = Some(i);
				},
				_ => return syn::Error::new_spanned(a, "invalid combination of CUDA function attributes").to_compile_error()
			}
		}
	}
	// Remove device_host_attr_index from attributes
	if let Some(i) = device_host {
		f.attrs.remove(i);
	}

	// Check if item is enabled
	match conv::is_item_enabled(&f.attrs, conv::CfgType::Declaration) {
		Ok(true) => {},
		Ok(false) => return TokenStream::new(), // Skip function
		Err(e) => return e.to_compile_error(),
	}

	// Process functions with both #[device] & #[host]
	let mut ts = TokenStream::new();
	if device_host.is_some() {
		if fn_type == Host {
			ts.extend(infallible_unwrap(process_device_fn(f.clone())))
		} else if fn_type == Device {
			ts.extend(infallible_unwrap(process_host_fn(f.clone())))
		}
	}

	match fn_type {
		Host => ts.extend(infallible_unwrap(process_host_fn(f))),
		Device => ts.extend(infallible_unwrap(process_device_fn(f))),
		Global => ts.extend(infallible_unwrap(process_global_fn(f))),
	}
	ts
}

fn process_all_fns_inner(item: syn::Item, fn_type: FunctionType, direct: bool) -> TokenStream {
	match item {
		syn::Item::Fn(f) => process_fn(f, fn_type),
		syn::Item::Mod(m) => {
			if let Some((_, items)) = m.content {
				let mut tt = TokenStream::new();
				for item in items {
					tt.append_all(process_all_fns_inner(item, fn_type, false))
				}
				tt
			} else {
				// TODO: Process other module somehow
				if direct {
					syn::Error::new_spanned(m, "expected function or inline module")
						.to_compile_error()
				} else {
					m.into_token_stream()
				}
			}
		},
		// TODO: syn::Item::Static => /* https://stackoverflow.com/questions/2619296/how-to-return-a-single-variable-from-a-cuda-kernel-function */,
		// TODO: syn::Item::Const => /* ... */,
		// TODO: syn::Item::Type => /* (type alias) */
		// TODO: syn::Item::Struct => /* ... */
		// TODO: syn::Item::Enum => /* ... */
		// TODO: syn::Item::Union => /* ... */
		_ => {
			if direct {
				syn::Error::new_spanned(item, "expected function or inline module")
					.to_compile_error()
			} else {
				quote!{ #item }
			}
		}
	}
}

fn process_all_fns(item: syn::Item, fn_type: FunctionType, direct: bool) -> TokenStream {
	use std::path::PathBuf;
	use std::fs::OpenOptions;
	use std::io::prelude::*;
	use std::process::{Command, Stdio};
	
	let out_path = std::env::var_os("CUDA_MACROS_OUT_DIR").map(|p| PathBuf::from(p).join("debug.rs"));
	let from_ts = if out_path.is_some() {
		let mut ts = TokenStream::new();
		item.to_tokens(&mut ts);
		Some(ts)
	} else {
		None
	};

	let ts = process_all_fns_inner(item, fn_type, direct);

	// Debug output
	if let Some(p) = out_path {
		if let Ok(mut f) = OpenOptions::new().read(true).append(true).create(true).open(&p) {
			writeln!(&mut f).ok();
			writeln!(&mut f, "/* *** FROM *** */").ok();
			writeln!(&mut f, "{}", &from_ts.unwrap()).ok();
			writeln!(&mut f, "/* ***  TO  *** */").ok();
			writeln!(&mut f, "{}", &ts).ok();
			writeln!(&mut f, "/* ************ */\n").ok();
			std::mem::drop(f);
			
			// Run rustfmt if possible
			Command::new("rustfmt").arg("--emit").arg("files").arg("--").arg(&p)
				.stdin(Stdio::null())
				.stdout(Stdio::null())
				.stderr(Stdio::null())
				.status().ok();
		}
	}
	ts
}


/// CUDA `__host__` function annotation
#[proc_macro_attribute]
pub fn host(attr: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream {
	let attr = TokenStream::from(attr);
	if !attr.is_empty() {
		return syn::Error::new_spanned(attr, "expected no attribute options")
			.to_compile_error().into();
	}
	let item = parse_macro_input!(item as syn::Item);
	process_all_fns(item, FunctionType::Host, true).into()
}

/// CUDA `__device__` function annotation
#[proc_macro_attribute]
pub fn device(attr: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream {
	let attr = TokenStream::from(attr);
	if !attr.is_empty() {
		return syn::Error::new_spanned(attr, "expected no attribute options")
			.to_compile_error().into();
	}
	let item = parse_macro_input!(item as syn::Item);
	process_all_fns(item, FunctionType::Device, true).into()
}

/// CUDA `__global__` function annotation
#[proc_macro_attribute]
pub fn global(attr: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream {
	let attr = TokenStream::from(attr);
	if !attr.is_empty() {
		return syn::Error::new_spanned(attr, "expected no attribute options")
			.to_compile_error().into();
	}
	let item = parse_macro_input!(item as syn::Item);
	process_all_fns(item, FunctionType::Global, true).into()
}
