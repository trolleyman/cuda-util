
use std::borrow::Cow;

use super::FunctionType;
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
			match (&ptr.mutability, &ptr.const_token) {
				(Some(m), Some(c)) => Err(Some(syn::Error::new_spanned(super::tokens_join2(m.clone(), c.clone()), "pointer is both const and mutable"))),
				(None, Some(_)) => Ok(format!("{}*", rust_type_to_c(&ptr.elem)?).into()),
				(Some(_), None) => Ok(format!("const {}*", rust_type_to_c(&ptr.elem)?).into()),
				(None, None) => Err(Some(syn::Error::new_spanned(ptr.clone(), "pointer must be mut or const"))),
			}
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
		Type::Infer(inner) => Err(Some(syn::Error::new_spanned(inner.clone(), "inferred types are not supported")))
	}
}

/// Converts a Rust type path into a C type
pub fn rust_type_path_to_c(type_path: &syn::TypePath) -> Result<Cow<'static, str>, Option<syn::Error>> {
	if let Some(q) = &type_path.qself {
		Err(Some(syn::Error::new_spanned(q.lt_token.clone(), "`<...>::type` style paths are not supported")))
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

#[derive(Copy, Clone, Debug)]
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
pub fn is_item_enabled_attr(attr: &syn::Attribute, _cfg_type: CfgType) -> Result<bool, syn::Error> {
	use syn::Meta;

	if FunctionType::try_from_attr(attr).is_some() {
		return Ok(true);
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
