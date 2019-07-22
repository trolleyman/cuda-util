
use std::borrow::Cow;

use super::conv;


const RUST_FIXED_NUM_TYPES: &'static [&'static str] = &["f32", "f64", "i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64"];
const C_FIXED_NUM_TYPES   : &'static [&'static str] = &["float", "double", "int8_t", "uint8_t", "int16_t", "uint16_t", "int32_t", "uint32_t", "int64_t", "uint64_t"];
const RUST_VAR_NUM_TYPES  : &'static [&'static str] = &["::std::os::raw::c_float", "::std::os::raw::c_double", "::std::os::raw::c_char", "::std::os::raw::c_schar", "::std::os::raw::c_uchar", "::std::os::raw::c_short", "::std::os::raw::c_ushort", "::std::os::raw::c_int", "::std::os::raw::c_uint", "::std::os::raw::c_long", "::std::os::raw::c_ulong", "::std::os::raw::c_longlong", "::std::os::raw::c_ulonglong"];
const C_VAR_NUM_TYPES     : &'static [&'static str] = &["float", "double", "char", "signed char", "unsigned char", "short", "unsigned short", "int", "unsigned int", "long", "unsigned long", "long long", "unsigned long long"];


/// Converts a Rust type into a C type
pub fn rust_type_to_c(ty: &syn::Type) -> Result<Cow<'static, str>, Option<syn::Error>> {
	use syn::Type;

	match &ty {
		Type::Slice(_) | Type::Array(_) => Err(None), // TODO: Auto-upload slices & arrays to CUDA
		Type::Ptr(ptr) => {
			let mutable = ptr.mutability.is_some();
			match (&ptr.mutability, &ptr.const_token) {
				(Some(m), Some(c)) => return Err(Some(syn::Error::new_spanned(super::tokens_join2(m.clone(), c.clone()), "pointer is both const and mutable"))),
				_ => {}
			}
			Ok(format!("{}*", rust_type_to_c(&ptr.elem)?).into())
		},
		Type::Reference(_) => Err(None), // TODO: Auto-uploading (?)
		Type::BareFn(_) | Type::Never(_) | Type::TraitObject(_) | Type::ImplTrait(_) | Type::Macro(_) | Type::Verbatim(_) => Err(None),
		Type::Tuple(t) if t.elems.len() == 0 => {
			// `()` type
			Ok("void".into())
		},
		Type::Tuple(_) => Err(None), // TODO: Auto-unpack & repack
		Type::Path(path) => rust_type_path_to_c(&path),
		Type::Paren(inner) => rust_type_to_c(&inner.elem),
		Type::Group(inner) => rust_type_to_c(&inner.elem),
		Type::Infer(inner) => Err(Some(syn::Error::new_spanned(inner.clone(), "arguments with inferred types are not allowed in CUDA function declarations")))
	}
}

/// Converts a Rust type path into a C type
pub fn rust_type_path_to_c(type_path: &syn::TypePath) -> Result<Cow<'static, str>, Option<syn::Error>> {
	if let Some(q) = &type_path.qself {
		Err(Some(syn::Error::new_spanned(q.lt_token.clone(), "`<...>::type` style paths in CUDA function declarations are not allowed")))
	} else {
		rust_path_to_c(&type_path.path)
	}
}

/// Converts a Rust path into a C type
pub fn rust_path_to_c(path: &syn::Path) -> Result<Cow<'static, str>, Option<syn::Error>> {
	// Check for number primitives
	for (&rty, &cty) in RUST_FIXED_NUM_TYPES.iter().zip(C_FIXED_NUM_TYPES.iter()) {
		if path.is_ident(rty) {
			return Ok(cty.into());
		}
	}
	// Check for ::std::os::raw::<c_num> types
	for (&rty, &cty) in RUST_VAR_NUM_TYPES.iter().zip(C_VAR_NUM_TYPES.iter()) {
		if super::type_path_matches(path, &rty) {
			return Ok(cty.into());
		}
	}
	if super::type_path_matches(path, "::std::ffi::c_void") || super::type_path_matches(path, "::core::ffi::c_void") {
		return Ok("void".into())
	}
	Err(None)
}

/// Converts a Rust function arg to a C arg
/// 
/// Returns `Ok(None)` when arg should be ignored
pub fn rust_fn_arg_to_c(arg: &syn::FnArg) -> Result<Option<(Cow<'static, str>, syn::Ident)>, syn::Error> {
	use syn::{FnArg, Pat};
	match arg {
		FnArg::SelfRef(arg) => Err(syn::Error::new_spanned(arg.clone(), "`self` is not allowed")),
		FnArg::SelfValue(arg) => Err(syn::Error::new_spanned(arg.clone(), "`self` is not allowed")),
		FnArg::Inferred(arg) => Err(syn::Error::new_spanned(arg.clone(), "arguments with inferred types are not allowed")),
		FnArg::Ignored(_) => Ok(None),
		FnArg::Captured(arg) => {
			match &arg.pat {
				Pat::Wild(_) => Ok(None),
				Pat::Ident(pat) if pat.by_ref.is_none() && pat.mutability.is_none() && pat.subpat.is_none() => {
					match conv::rust_type_to_c(&arg.ty) {
						Ok(t) => Ok(Some((t, pat.ident.clone()))),
						Err(Some(e)) => Err(e),
						Err(None) => Err(syn::Error::new_spanned(arg.ty.clone(), "arguments of this type are not allowed")),
					}
				},
				_ => Err(syn::Error::new_spanned(arg.pat.clone(), "only simple bound arguments are allowed")),
			}
		}
	}
}
