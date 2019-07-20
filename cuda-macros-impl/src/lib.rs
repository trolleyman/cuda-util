
extern crate proc_macro;

use proc_macro2::TokenStream;
use syn::parse_macro_input;
use quote::{quote, ToTokens};
use quote::TokenStreamExt;

use cuda_macros_util::FunctionType;


fn check_function_signature(f: &syn::ItemFn) -> Option<TokenStream> {
	if f.attrs.len() != 0 {
		return Some(syn::Error::new_spanned(f.attrs[0].clone(), "attributes on CUDA functions are not supported")
			.to_compile_error().into())
	}
	if let Some(item) = &f.constness {
		return Some(syn::Error::new_spanned(item.clone(), "const functions are not supported")
			.to_compile_error().into())
	}
	if let Some(item) = &f.asyncness {
		return Some(syn::Error::new_spanned(item.clone(), "async functions are not supported")
			.to_compile_error().into())
	}
	if let Some(item) = &f.decl.variadic {
		return Some(syn::Error::new_spanned(item.clone(), "varadic functions are not supported")
			.to_compile_error().into())
	}
	if f.decl.generics.params.len() != 0 {
		return Some(syn::Error::new_spanned(f.decl.generics.params.clone(), "generic functions are not supported")
			.to_compile_error().into())
	}
	if let Some(item) = &f.decl.generics.where_clause {
		return Some(syn::Error::new_spanned(item.clone(), "generic functions are not supported")
			.to_compile_error().into())
	}
	None
}

fn process_host_fn(f: syn::ItemFn) -> TokenStream {
	// Pass through
	// TODO: handle cfg(__CUDA_ARCH__)
	// https://stackoverflow.com/questions/23899190/can-a-host-device-function-know-where-it-is-executing
	// https://docs.nvidia.com/cuda/archive/9.0/cuda-c-programming-guide/index.html#host
	quote!{ #f }
}

fn process_device_fn(_f: syn::ItemFn) -> TokenStream {
	// Output nothing - already compiled device version in build.rs
	TokenStream::new()
}

fn process_global_fn(f: syn::ItemFn) -> TokenStream {
	// Create function wrapper & link to CUDA function
	if f.unsafety.is_none() {
		return syn::Error::new_spanned(f, "#[global] functions are required to be marked as unsafe")
			.to_compile_error().into()
	}

	let ident = f.ident.clone();
	let c_ident = quote!{ _rust_cuda_macros_cwrapper_#ident };
	let vis = f.vis;
	let args = unimplemented!();
	let c_args_decl = unimplemented();
	let args_decl = f.decl.inputs;
	let ret = f.decl.output;
	quote!{
		extern "C" fn #c_ident(#c_args_decl);

		unsafe #vis fn #ident(config: impl ::std::convert::Into<::cuda_macros::ExecutionConfig>, #args_decl) -> #ret {
			let config = config.into();
			#c_ident(, #args)
		}
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
				_ => return syn::Error::new_spanned(a, "invalid combination of CUDA function attributes")
						.to_compile_error().into()
			}
		}
	}
	// Remove device_host_attr_index from attributes
	if let Some(i) = device_host {
		f.attrs.remove(i);
	}

	// Check function signatures
	if let Some(err) = check_function_signature(&f) {
		return err;
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
