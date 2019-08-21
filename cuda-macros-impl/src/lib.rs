#![doc(html_logo_url = "https://raw.githubusercontent.com/trolleyman/cuda-util/master/cuda-macros/res/logo_512.png")]
//! __This is a private crate, meant to only be used by [`cuda-macros-impl`](https://docs.rs/cuda-macros-impl).__
//! 
//! __Do NOT re-export anywhere else, as this messes up the paths emitted by the macros in
//! [`cuda-macros-impl`](https://docs.rs/cuda-macros-impl).__

extern crate cuda_macros_common as util;

extern crate proc_macro;
extern crate cuda;
extern crate proc_macro2;
extern crate syn;
extern crate quote;
extern crate fs2;

mod output;

use proc_macro2::{TokenStream, Span};
use syn::parse_macro_input;
use quote::{quote, ToTokens};
use quote::TokenStreamExt;

use util::{conv, FunctionType, FnInfo, GpuTypeEnum};


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

fn process_global_fn(f: syn::ItemFn) -> Result<TokenStream, TokenStream> {
	// Output function
	let fn_info = FnInfo::new(&f, FunctionType::Global).map_err(|e| e.to_compile_error())?;
	output::output_fn(&f, &fn_info)?;

	// Create function wrapper & link to CUDA function
	use syn::{FnArg, Pat};

	if f.sig.unsafety.is_none() {
		return Err(syn::Error::new_spanned(f, "#[global] functions are required to be marked as unsafe").to_compile_error());
	}

	let args_decl = f.sig.inputs;
	let mut args = vec![];
	for arg in args_decl.iter() {
		match arg {
			FnArg::Typed(arg) => {
				let ident = match &*arg.pat {
					Pat::Wild(_) => continue,
					Pat::Ident(item) => item.ident.clone(),
					_ => panic!("*should* never happen due to `check_function_signature`")
				};
				args.push(ident);
			},
			FnArg::Receiver(_) => continue,
		}
	}

	let vis = f.vis;
	let ret = f.sig.output;
	let generics_params = f.sig.generics.params;

	let fn_ident = f.sig.ident.clone();
	let fn_ident_c_base = format!("rust_cuda_macros_wrapper_{}_{}", fn_info.number_generics.len(), fn_ident);

	if !fn_info.is_generic() {
		let args = quote!{ #(#args),* };
		let fn_ident_c = syn::Ident::new(&fn_ident_c_base, fn_ident.span());

		let extern_c_fn = quote!{
			extern "C" {
				#vis fn #fn_ident_c(config: ::cuda_macros::ExecutionConfig, #args_decl) #ret;
			}
		};

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
		let mut extern_fns = TokenStream::new();

		// Generate `extern "C"` functions
		let mut fn_ident_c = String::with_capacity(fn_ident_c_base.len() + 3 * fn_info.number_generics.len());
		let mut tys_extra = vec![];
		cuda_macros_common::permutations(GpuTypeEnum::types(), fn_info.number_generics.len(), |tys| {
			// Get correct ident name (e.g. compare_f32_u32 is from compare<T: GpuType, U: GpuType>(x: T, y: T) with f32 and u32 type args)
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
			let args_decl_instantiated = util::instantiate_generic_args(&args_decl, &tys_extra[..]);
			extern_fns.extend(quote!{
				#vis fn #fn_ident_c(config: ::cuda_macros::ExecutionConfig, #args_decl_instantiated) #ret;
			});
		});
		let extern_fns = quote!{
			extern "C" {
				#extern_fns
			}
		};

		// <T: GpuType, etc.>
		let mut generics_decl = TokenStream::new();
		if let Some(lt_token) = &f.sig.generics.lt_token {
			generics_decl.extend(quote!{ #lt_token });
		} else {
			generics_decl.extend(quote!{ < });
		}
		generics_decl.extend(quote!{ #generics_params });
		if let Some(gt_token) = &f.sig.generics.gt_token {
			generics_decl.extend(quote!{ #gt_token });
		} else {
			generics_decl.extend(quote!{ > });
		}

		// where T: GpuType
		let mut where_clause = TokenStream::new();
		if let Some(clause) = f.sig.generics.where_clause {
			where_clause.extend(quote!{ #clause });
		}
		
		// Wrapper function body
		let args_expr: Vec<_> = args.iter().map(|ident| quote!{ ::std::mem::transmute(#ident) }).collect();
		let args_expr = quote!{ #(#args_expr),* };
		
		// Get vector of (generic type arg name, identifier)
		let mut match_pattern_types = vec![];
		for ty_arg_name in fn_info.number_generics.iter() {
			for arg in args_decl.iter() {
				match arg {
					FnArg::Typed(arg) => {
						let ident = match &*arg.pat {
							Pat::Wild(_) => continue,
							Pat::Ident(item) => &item.ident,
							_ => panic!("*Should* never happen due to `check_function_signature`")
						};
						// Check for `x: T` args
						if let Ok(p) = util::get_type_path(&arg.ty) {
							if p.is_ident(ty_arg_name) {
								match_pattern_types.push((ty_arg_name.clone(), ident.clone()));
							}
						}
					},
					FnArg::Receiver(_) => continue,
				}
			}
		}
		
		let mut match_inputs = vec!{};
		for (_, ident) in match_pattern_types.iter() {
			match_inputs.push(quote!{
				::cuda_macros::GpuType::gpu_type(&#ident)
			});
		}
		let match_input = quote!{ (#(#match_inputs,)*) };

		let mut match_inner = quote!{};
		cuda_macros_common::permutations(GpuTypeEnum::types(), fn_info.number_generics.len(), |tys| {
			// Generate match arm pattern
			let mut match_patterns = vec!{};
			for (ty, ident) in match_pattern_types.iter() {
				let i = fn_info.number_generics.iter().position(|t| t == ty).unwrap_or(0);
				let enum_name = tys[i].enum_name();
				let enum_name = syn::Ident::new(&enum_name, Span::call_site());
				match_patterns.push(quote!{
					::cuda_macros::GpuTypeEnum::#enum_name(#ident)
				});
			}
			let match_pattern = quote!{ (#(#match_patterns,)*) };
			
			// Generate match arm value
			fn_ident_c.clone_from(&fn_ident_c_base);
			for ty in tys.iter() {
				fn_ident_c.push_str("_");
				fn_ident_c.push_str(ty.rust_name());
			}
			let fn_ident_c = syn::Ident::new(&fn_ident_c, fn_ident.span());
			
			let match_expr = quote!{
				#fn_ident_c(config, #args_expr)
			};
			
			match_inner.extend(quote!{
				#match_pattern => #match_expr,
			});
		});

		let wrapper_fn_body = quote!{
			let config = config.into();
			match #match_input {
				#match_inner
				_ => unreachable!("report this bug to https://github.com/trolleyman/cuda-util with RUST_BACKTRACE=1"),
			}
		};

		let wrapper_fn = quote!{
			#vis unsafe fn #fn_ident#generics_decl(config: impl ::std::convert::Into<::cuda_macros::ExecutionConfig>, #args_decl) #ret #where_clause {
				#wrapper_fn_body
			}
		};

		Ok(quote!{
			#extern_fns
			#wrapper_fn
		})
	}
}

fn infallible_unwrap<T>(val: Result<T, T>) -> T {
	match val {
		Ok(t) => t,
		Err(t) => t,
	}
}

fn process_fn(mut f: syn::ItemFn, fn_type: FunctionType) -> TokenStream {
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

fn try_rustfmt(data: &[u8]) -> Result<Vec<u8>, ()> {
	use std::process::{Command, Stdio};
	use std::io::prelude::*;

	let mut child = Command::new("rustfmt")
		.stdin(Stdio::piped())
		.stdout(Stdio::piped())
		.stderr(Stdio::null())
		.spawn()
		.map_err(|_| ())?;
	
	child.stdin.as_mut().unwrap().write_all(&data).map_err(|_| ())?;
	let output = child.wait_with_output().map_err(|_| ())?;
	if output.status.success() {
		Ok(output.stdout)
	} else {
		Err(())
	}
}

fn process_all_fns(item: syn::Item, fn_type: FunctionType, direct: bool) -> TokenStream {
	use std::path::PathBuf;
	use std::fs::OpenOptions;
	use std::io::prelude::*;
	use std::io::Cursor;
	
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
			let mut data = Cursor::new(Vec::new());
			writeln!(&mut data, "/* *** FROM *** */").ok();
			writeln!(&mut data, "{}", &from_ts.unwrap()).ok();
			writeln!(&mut data, "/* ***  TO  *** */").ok();
			writeln!(&mut data, "{}", &ts).ok();
			writeln!(&mut data, "/* ************ */\n").ok();
			
			// Run rustfmt if possible
			let data = try_rustfmt(&data.get_ref()).unwrap_or(data.into_inner());
			f.write_all(&data).ok();
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
