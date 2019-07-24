
extern crate proc_macro;
extern crate cuda_macros_util;
extern crate cuda;
extern crate proc_macro2;
extern crate syn;
extern crate quote;
extern crate fs2;

mod output;


use proc_macro2::TokenStream;
use syn::parse_macro_input;
use quote::{quote, ToTokens};
use quote::TokenStreamExt;

use cuda_macros_util::FunctionType;


fn check_function_signature(f: &syn::ItemFn) -> Result<(), syn::Error> {
	if f.attrs.len() != 0 {
		// TODO: #[include(cstdio)], #[cfg(<soemthing>)], etc.
		return Err(syn::Error::new_spanned(f.attrs[0].clone(), "attributes on CUDA functions are not allowed"));
	}
	if let Some(item) = &f.constness {
		return Err(syn::Error::new_spanned(item.clone(), "const CUDA functions are not allowed"));
	}
	if let Some(item) = &f.asyncness {
		return Err(syn::Error::new_spanned(item.clone(), "async CUDA functions are not allowed"));
	}
	if let Some(item) = &f.abi {
		return Err(syn::Error::new_spanned(item.clone(), "non-default ABIs on CUDA functions are not allowed"));
	}
	if let Some(item) = &f.decl.variadic {
		return Err(syn::Error::new_spanned(item.clone(), "varadic CUDA functions are not allowed"));
	}
	if f.decl.generics.params.len() != 0 {
		return Err(syn::Error::new_spanned(f.decl.generics.params.clone(), "generic CUDA functions are not allowed"));
	}
	if let Some(item) = &f.decl.generics.where_clause {
		return Err(syn::Error::new_spanned(item.clone(), "generic CUDA functions are not allowed"));
	}
	Ok(())
}

fn process_host_fn(f: syn::ItemFn) -> TokenStream {
	// Pass through
	// TODO: handle cfg!(__CUDA_ARCH__)
	// https://stackoverflow.com/questions/23899190/can-a-host-device-function-know-where-it-is-executing
	// https://docs.nvidia.com/cuda/archive/9.0/cuda-c-programming-guide/index.html#host
	quote!{ #f }
}

fn process_device_fn(f: syn::ItemFn) -> TokenStream {
	// Output function
	if let Err(ts) = output::output_fn(&f, FunctionType::Device) {
		return ts;
	}

	let ident = f.ident;
	let vis = f.vis;

	// Return dummy identifier that will cause a compilation error when called
	// TODO: give a better compile time error somehow (maybe custom type?)
	quote!{
		#vis const #ident: () = ();
	}
}

fn process_global_fn(f: syn::ItemFn) -> TokenStream {
	// Output function
	if let Err(ts) = output::output_fn(&f, FunctionType::Global) {
		return ts;
	}

	// Create function wrapper & link to CUDA function
	use syn::{FnArg, Pat};

	if f.unsafety.is_none() {
		return syn::Error::new_spanned(f, "#[global] functions are required to be marked as unsafe").to_compile_error();
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

	let fn_ident = f.ident.clone();
	let fn_ident_c = format!("rust_cuda_macros_cwrapper_{}", fn_ident);
	let fn_ident_c = syn::Ident::new(&fn_ident_c, fn_ident.span());

	let vis = f.vis;
	let ret = f.decl.output;

	let extern_c_fn = quote!{ extern "C" { #vis fn #fn_ident_c(config: ::cuda_macros::ExecutionConfig, #args_decl) #ret; } };

	let wrapper_fn = quote!{
		unsafe #vis fn #fn_ident(config: impl ::std::convert::Into<::cuda_macros::ExecutionConfig>, #args_decl) #ret {
			let config = config.into();
			#fn_ident_c(config, #args)
		}
	};

	quote!{
		#extern_c_fn
		#wrapper_fn
	}
}

fn process_fn(mut f: syn::ItemFn, fn_type: FunctionType) -> TokenStream {
	use cuda_macros_util::FunctionType::*;

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

	// Check function signatures
	if let Err(e) = check_function_signature(&f) {
		return e.to_compile_error();
	}

	// Process functions with both #[device] & #[host]
	let mut ts = TokenStream::new();
	if device_host.is_some() && fn_type == Host {
		if fn_type == Host {
			ts.extend(process_device_fn(f.clone()))
		} else if fn_type == Device {
			ts.extend(process_host_fn(f.clone()))
		}
	}

	match fn_type {
		Host => ts.extend(process_host_fn(f)),
		Device => ts.extend(process_device_fn(f)),
		Global => ts.extend(process_global_fn(f)),
	}
	ts
}

fn process_all_fns(item: syn::Item, fn_type: FunctionType, direct: bool) -> TokenStream {
	match item {
		syn::Item::Fn(f) => process_fn(f, fn_type),
		syn::Item::Mod(m) => {
			if let Some((_, items)) = m.content {
				let mut tt = TokenStream::new();
				for item in items {
					tt.append_all(process_all_fns(item, fn_type, false))
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
