
use std::borrow::Cow;

use super::{FunctionType, FnInfo};
use super::conv;


const RUST_FIXED_NUM_TYPES: &'static [&'static str] = &["f32", "f64", "i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64", "usize"];  // TODO: isize
const C_FIXED_NUM_TYPES   : &'static [&'static str] = &["float", "double", "int8_t", "uint8_t", "int16_t", "uint16_t", "int32_t", "uint32_t", "int64_t", "uint64_t", "size_t"];
const RUST_VAR_NUM_TYPES  : &'static [&'static str] = &["::std::os::raw::c_float", "::std::os::raw::c_double", "::std::os::raw::c_char", "::std::os::raw::c_schar", "::std::os::raw::c_uchar", "::std::os::raw::c_short", "::std::os::raw::c_ushort", "::std::os::raw::c_int", "::std::os::raw::c_uint", "::std::os::raw::c_long", "::std::os::raw::c_ulong", "::std::os::raw::c_longlong", "::std::os::raw::c_ulonglong"];
const C_VAR_NUM_TYPES     : &'static [&'static str] = &["float", "double", "char", "signed char", "unsigned char", "short", "unsigned short", "int", "unsigned int", "long", "unsigned long", "long long", "unsigned long long"];
const SHORTHAND_TYPES     : &'static [&'static str] = &["float", "double", "char", "schar", "uchar", "short", "ushort", "int", "uint", "long", "ulong", "longlong", "ulonglong"];


fn rust_shared_type_to_c_inner(ty: &syn::Type, fn_info: &FnInfo) -> Result<(Cow<'static, str>, Cow<'static, str>), Option<syn::Error>> {
	use syn::{Type, Expr};
	match ty {
		Type::Slice(ty) => Ok((format!("extern __shared__ {}", rust_type_to_c(&ty.elem, fn_info, false)?).into(), "[]".into())),
		Type::Array(ty) => match &ty.len {
			Expr::Lit(syn::ExprLit{ lit: syn::Lit::Int(lit), .. }) => 
				Ok((format!("__shared__ {}", rust_type_to_c(&ty.elem, fn_info, false)?).into(), format!("[{}]", lit.value()).into())),
			_ => Err(Some(syn::Error::new_spanned(ty.clone(), "array length must be a literal number constant")))
		},
		Type::Reference(ty) => match &*ty.elem {
			Type::Reference(ty) => Err(Some(syn::Error::new_spanned(ty.clone(), "reference-to-reference types are not supported in a #[shared] declaration"))),
			Type::Array(_) => Err(None),
			_ => rust_shared_type_to_c_inner(&ty.elem, fn_info)
		},
		Type::Paren(ty) => rust_shared_type_to_c_inner(&ty.elem, fn_info),
		Type::Group(ty) => rust_shared_type_to_c_inner(&ty.elem, fn_info),
		Type::Infer(ty) => Err(Some(syn::Error::new_spanned(ty.clone(), "inferred types are not supported"))),
		_=> Err(None),
	}
}


/// Converts a Rust `#[shared]` type into a C type
/// 
/// Returns `(prefix, postfix)` on success. 
pub fn rust_shared_type_to_c(ty: &syn::Type, fn_info: &FnInfo) -> Result<(Cow<'static, str>, Cow<'static, str>), syn::Error> {
	rust_shared_type_to_c_inner(ty, fn_info)
		.map_err(|e| e.unwrap_or_else(|| syn::Error::new_spanned(ty.clone(), "only slice/array types are supported in a #[shared] declaration")))
}

/// Converts a Rust type into a C type
pub fn rust_type_to_c(ty: &syn::Type, fn_info: &FnInfo, cnst: bool) -> Result<Cow<'static, str>, Option<syn::Error>> {
	use syn::Type;

	let cty = match &ty {
		Type::Slice(ty) => Err(Some(syn::Error::new_spanned(ty.clone(), "slice types are not yet supported"))), // TODO: slices
		Type::Array(ty) => Err(Some(syn::Error::new_spanned(ty.clone(), "array types are not yet supported"))), // TODO: arrays
		Type::Ptr(ty) => {
			match (&ty.mutability, &ty.const_token) {
				(Some(m), Some(c)) => Err(Some(syn::Error::new_spanned(super::tokens_join2(m.clone(), c.clone()), "pointer is both const and mutable"))),
				(None, Some(_)) => Ok(format!("{}* const", rust_type_to_c(&ty.elem, fn_info, true)?).into()),
				(Some(_), None) => Ok(format!("{}*", rust_type_to_c(&ty.elem, fn_info, false)?).into()),
				(None, None) => Err(Some(syn::Error::new_spanned(ty.clone(), "pointer must be mut or const"))),
			}
		},
		Type::Reference(ty) => Err(Some(syn::Error::new_spanned(ty.clone(), "reference types are not yet supported"))), // TODO: references
		Type::BareFn(_) | Type::Never(_) | Type::TraitObject(_) | Type::ImplTrait(_) | Type::Macro(_) | Type::Verbatim(_) => Err(None),
		Type::Tuple(ty) if ty.elems.len() == 0 => {
			// `()` type
			Ok("void".into())
		},
		Type::Tuple(ty) => Err(Some(syn::Error::new_spanned(ty.clone(), "tuple types are not yet supported"))), // TODO: tuples
		Type::Path(path) => rust_type_path_to_c(&path, fn_info),
		Type::Paren(ty) => rust_type_to_c(&ty.elem, fn_info, cnst),
		Type::Group(ty) => rust_type_to_c(&ty.elem, fn_info, cnst),
		Type::Infer(ty) => Err(Some(syn::Error::new_spanned(ty.clone(), "inferred types are not supported")))
	};
	if cnst {
		cty.map(|c| format!("{} const", c).into())
	} else {
		cty
	}
}

/// Converts a Rust type path into a C type
pub fn rust_type_path_to_c(type_path: &syn::TypePath, fn_info: &FnInfo) -> Result<Cow<'static, str>, Option<syn::Error>> {
	if let Some(q) = &type_path.qself {
		Err(Some(syn::Error::new_spanned(q.lt_token.clone(), "`<...>::type` style paths are not supported")))
	} else {
		rust_path_to_c(&type_path.path, fn_info)
	}
}

/// Converts a Rust path into a C type
pub fn rust_path_to_c(path: &syn::Path, fn_info: &FnInfo) -> Result<Cow<'static, str>, Option<syn::Error>> {
	use proc_macro2::TokenStream;
	use quote::ToTokens;

	// Check for Rust number primitives
	for (&rty, &cty) in RUST_FIXED_NUM_TYPES.iter().zip(C_FIXED_NUM_TYPES.iter()) {
		if path.is_ident(rty) {
			return Ok(cty.into());
		}
	}

	// Check for shorthand primitive types (float, uint, int, longlong, etc.)
	for (&sty, &cty) in SHORTHAND_TYPES.iter().zip(C_VAR_NUM_TYPES.iter()) {
		if path.is_ident(sty) {
			return Ok(cty.into());
		}
	}

	// Check for ::std::os::raw::<c_num> types
	for (&rty, &cty) in RUST_VAR_NUM_TYPES.iter().zip(C_VAR_NUM_TYPES.iter()) {
		if super::type_path_matches(path, &rty) {
			return Ok(cty.into());
		}
	}

	// Check for void
	if super::type_path_matches(path, "::std::ffi::c_void") || super::type_path_matches(path, "::core::ffi::c_void") {
		return Ok("void".into())
	}

	// Check for generics
	for generic_name in fn_info.number_generics.iter() {
		if path.is_ident(generic_name) {
			return Ok(generic_name.clone().into());
		}
	}

	// If it is an unknown ident, just pass through
	// TODO: Figure out a better way of doing this
	if path.leading_colon.is_none() && path.segments.len() == 1 {
		let mut ts = TokenStream::new();
		path.to_tokens(&mut ts);
		return Ok(format!("{}", ts).into())
	}
	Err(None)
}

/// Converts a Rust function arg to a C arg
/// 
/// Returns `Ok(None)` when arg should be ignored
pub fn rust_fn_arg_to_c(arg: &syn::FnArg, fn_info: &FnInfo) -> Result<Option<(Cow<'static, str>, syn::Ident)>, syn::Error> {
	use syn::{FnArg, Pat};
	match arg {
		FnArg::SelfRef(arg) => Err(syn::Error::new_spanned(arg.clone(), "`self` is not allowed")),
		FnArg::SelfValue(arg) => Err(syn::Error::new_spanned(arg.clone(), "`self` is not allowed")),
		FnArg::Inferred(arg) => Err(syn::Error::new_spanned(arg.clone(), "arguments with inferred types are not allowed")),
		FnArg::Ignored(_) => Ok(None),
		FnArg::Captured(arg) => {
			match &arg.pat {
				Pat::Wild(_) => Ok(None),
				Pat::Ident(pat) if pat.by_ref.is_none() && pat.subpat.is_none() => {
					match conv::rust_type_to_c(&arg.ty, fn_info, pat.mutability.is_none()) {
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CfgType {
	Declaration,
	DeviceCode,
	HostCode,
}

/// Check if all of the attributes are valid and mean that the item/expression it is attached to is enabled
pub fn is_item_enabled<'a, I: IntoIterator<Item=&'a syn::Attribute>>(attrs: I, cfg_type: CfgType) -> Result<bool, syn::Error> {
	for attr in attrs.into_iter() {
		if !is_item_enabled_attr(attr, cfg_type)? {
			return Ok(false);
		}
	}
	Ok(true)
}

/// Check if an attribute is valid and means that the item/expression it is attached to is enabled
pub fn is_item_enabled_attr(attr: &syn::Attribute, cfg_type: CfgType) -> Result<bool, syn::Error> {
	use syn::Meta;

	if FunctionType::try_from_attr(attr).is_some() {
		return Ok(true);
	}

	if cfg_type == CfgType::DeviceCode {
		if attr.path.is_ident("shared") {
			return Ok(true);
		}
	}

	if let syn::AttrStyle::Inner(_) = attr.style {
		return Err(syn::Error::new_spanned(attr.clone(), "inner attributes are not supported"));
	}

	if super::type_path_matches(&attr.path, "::cfg") {
		match attr.parse_meta() {
			Ok(Meta::Word(meta)) => return Err(syn::Error::new_spanned(meta.clone(), "no arguments specified to cfg")),
			Ok(Meta::List(_meta)) => {
				// TODO
				unimplemented!("#[cfg(<option>)]");
			},
			Ok(Meta::NameValue(meta)) => return Err(syn::Error::new_spanned(meta.clone(), "cfg meta syntax not supported")),
			Err(_) => return Err(syn::Error::new_spanned(attr.clone(), "cfg meta syntax not supported"))
		}
	}
	Err(syn::Error::new_spanned(attr.path.clone(), "unsupported attribute"))
}
