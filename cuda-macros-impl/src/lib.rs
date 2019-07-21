
extern crate proc_macro;

use proc_macro2::TokenStream;
use syn::parse_macro_input;
use quote::{quote, quote_spanned, ToTokens};
use quote::TokenStreamExt;

use cuda_macros_util::{type_path_matches, FunctionType};


const RUST_NUM_TYPES: &'static [&'static str] = &["f32", "f64", "u8", "i8", "u16", "i16", "u32", "i32", "u64", "i64"];
const RUST_C_NUM_TYPES: &'static [&'static str] = &["c_char", "c_double", "c_float", "c_int", "c_long", "c_longlong", "c_schar", "c_short", "c_uchar", "c_uint", "c_ulong", "c_ulonglong", "c_ushort"];

fn is_type_path_disallowed(type_path: &syn::TypePath) -> (bool, Option<TokenStream>) {
	if let Some(q) = &type_path.qself {
		return (true, Some(syn::Error::new_spanned(q.clone(), "arguments with complicated types are not allowed in CUDA function declarations")
			.to_compile_error().into()))
	}
	for ty in RUST_NUM_TYPES.iter() {
		if type_path.is_ident(*ty) {
			return (false, None);
		}
	}
	for ty in RUST_C_NUM_TYPES.iter() {
		if type_path.is_ident(*ty) {
			return (false, None);
		}
	}

	// Check for ::std::os::raw::<c_num> types
	for ty in RUST_C_NUM_TYPES.iter().map(|ty| format!("::std::os::raw::{}", ty)) {
		if type_path_matches(&type_path, &ty) {
			return (false, None);
		}
	}
	(true, None)
}

fn is_type_disallowed(ty: &syn::Type) -> (bool, Option<TokenStream>) {
	match &ty {
		Type::Slice(_) | Type::Array(_) => (true, None), // TODO: Auto-upload slices & arrays to CUDA
		Type::Ptr(_) => (false, None),
		Type::Reference(_) => (true, None), // TODO: Auto-uploading (?)
		Type::BareFn(_) | Type::Never(_) | Type::TraitObject(_) | Type::ImplTrait(_) | Type::Paren | Type::Infer(_) | Type::Macro(_) | Type::Verbatim(_) => (true, None),
		Type::Tuple(_) => (true, None), // TODO: Auto-unpack & repack
		Type::Path(path) => is_type_path_disallowed(&path),
		Type::Pren(inner) => is_type_disallowed(&inner),
		Type::Group(inner) => is_type_disallowed(&inner.elem),
		Type::Infer(inner) => return (true, Some(syn::Error::new_spanned(inner.clone(), "arguments with inferred types are not allowed in CUDA function declarations")
			.to_compile_error().into()));
	}
}

fn check_function_signature(f: &syn::ItemFn) -> Option<TokenStream> {
	use syn::{FnArg, Pat, Type};

	if f.attrs.len() != 0 {
		return Some(syn::Error::new_spanned(f.attrs[0].clone(), "attributes on CUDA functions are not allowed")
			.to_compile_error().into())
	}
	if let Some(item) = &f.constness {
		return Some(syn::Error::new_spanned(item.clone(), "const CUDA functions are not allowed")
			.to_compile_error().into())
	}
	if let Some(item) = &f.asyncness {
		return Some(syn::Error::new_spanned(item.clone(), "async CUDA functions are not allowed")
			.to_compile_error().into())
	}
	if let Some(item) = &f.decl.variadic {
		return Some(syn::Error::new_spanned(item.clone(), "varadic CUDA functions are not allowed")
			.to_compile_error().into())
	}
	if f.decl.generics.params.len() != 0 {
		return Some(syn::Error::new_spanned(f.decl.generics.params.clone(), "generic CUDA functions are not allowed")
			.to_compile_error().into())
	}
	if let Some(item) = &f.decl.generics.where_clause {
		return Some(syn::Error::new_spanned(item.clone(), "generic CUDA functions are not allowed")
			.to_compile_error().into())
	}
	
	// Check argument spec
	for arg in f.decl.inputs.iter() {
		match arg {
			FnArg::SelfRef(arg) | FnArg::SelfValue(arg) => return Some(syn::Error::new_spanned(arg.clone(), format!("`{}` is not allowed in CUDA function declarations", &arg))
				.to_compile_error().into()),
			FnArg::Inferred(arg) => return Some(syn::Error::new_spanned(arg.clone(), "arguments with inferred types are not allowed in CUDA function declarations")
				.to_compile_error().into()),
			FnArg::Ignored(_) => {},
			FnArg::Captured(arg) => {
				let disallowed = match &arg.pat {
					Pat::Wild(_) => false,
					Pat::Ident(pat) => pat.by_ref.is_some() || pat.mutability.is_some() || pat.subpat.is_some(),
					_ => true,
				};
				if disallowed {
					return Some(syn::Error::new_spanned(arg.pat.clone(), "only simple bound arguments are allowed in CUDA function declarations")
						.to_compile_error().into());
				}
				let disallowed_ty = is_type_disallowed(&arg.ty);
				if let Some(ty) = disallowed_ty {
					return Some(syn::Error::new_spanned(arg.ty.clone(), format!("arguments of type `{}` are not allowed in CUDA function declarations", &arg.ty))
						.to_compile_error().into());
				}
			}
		}
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

	let args_decl = f.decl.inputs;
	for arg in args_decl.iter() {
		match arg {
			SelfRef(item) => return syn::Error::new_spanned(f, "self not allowed in ")
				.to_compile_error().into(),
		}
	}
	let args = unimplemented!();
	let args_c_decl = unimplemented();

	let fn_ident = f.ident.clone();
	let fn_ident_c = quote!{ _rust_cuda_macros_cwrapper_#fn_ident };

	let vis = f.vis;
	let ret = f.decl.output;

	quote!{
		extern "C" fn #fn_ident_c(#args_c_decl) -> #ret_c;

		unsafe #vis fn #fn_ident(config: impl ::std::convert::Into<::cuda_macros::ExecutionConfig>, #args_decl) -> #ret {
			let config = config.into();
			let ret = #fn_ident_c(, #args);
			Into::<#ret>::into(ret)
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
