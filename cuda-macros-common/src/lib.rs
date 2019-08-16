
extern crate cuda;
extern crate proc_macro2;
extern crate syn;
extern crate quote;
extern crate chrono;

use syn::spanned::Spanned;
use proc_macro2::TokenStream;
use quote::ToTokens;

pub mod conv;
pub mod write;
pub mod file;
mod perms;
mod execution_config;
mod cuda_number;

pub use perms::*;
pub use execution_config::{Dim3, ExecutionConfig};
pub use cuda_number::{CudaNumber, CudaNumberType};


#[derive(Clone, Debug)]
pub struct FnInfo {
	pub ty: FunctionType,
	pub number_generics: Vec<String>,
	pub c_template_decl: String,
}
impl FnInfo {
	/// Get information about the generics of a function, and report an error if invalid
	pub fn new(f: &syn::ItemFn, fn_type: FunctionType) -> Result<FnInfo, syn::Error> {
		use syn::{punctuated::Punctuated, token::Add, GenericParam, TypeParamBound, TraitBoundModifier, WherePredicate};

		fn type_param_bound_is_cuda_number(bounds: &Punctuated<TypeParamBound, Add>) -> Result<bool, syn::Error> {
			if bounds.len() == 0 {
				Ok(false)
			} else if bounds.len() > 1 {
				Err(syn::Error::new_spanned(bounds, "generic param must be bound by `CudaNumber`, e.g. `T: CudaNumber`"))
			} else {
				match &bounds[0] {
					TypeParamBound::Trait(bound) => {
						if bound.modifier != TraitBoundModifier::None {
							Err(syn::Error::new_spanned(bound.modifier, "generic param must be bound by `CudaNumber`, e.g. `T: CudaNumber`"))
						} else if let Some(lts) = &bound.lifetimes {
							Err(syn::Error::new_spanned(lts, "generic param must be bound by `CudaNumber`, e.g. `T: CudaNumber`"))
						} else if type_path_matches(&bound.path, "::cuda_macros_common::CudaNumber") ||
								type_path_matches(&bound.path, "::cuda_macros::CudaNumber") ||
								type_path_matches(&bound.path, "::cuda_util::CudaNumber") {
							Ok(true)
						} else {
							Err(syn::Error::new_spanned(&bound.path, "generic param must be bound by `CudaNumber`, e.g. `T: CudaNumber`"))
						}
					},
					TypeParamBound::Lifetime(bound) => {
						Err(syn::Error::new_spanned(bound, "generic param must be bound by `CudaNumber`, e.g. `T: CudaNumber`"))
					}
				}
			}
		}

		let mut generics: Vec<(syn::Ident, proc_macro2::Span, bool)> = Vec::new();

		// Get `T: CudaNumber` generics
		for generic in &f.decl.generics.params {
			match generic {
				GenericParam::Type(generic) => {
					if conv::is_item_enabled(&generic.attrs, conv::CfgType::DeviceCode)? {
						if let Some(def) = &generic.default {
							return Err(syn::Error::new_spanned(def, "default generic parameters not supported"));
						}
						if generics.iter().find(|(ident, _, _)| *ident == generic.ident).is_some() {
							return Err(syn::Error::new_spanned(&generic.ident, "generic already declared"));
						}
						generics.push((generic.ident.clone(), generic.span(), type_param_bound_is_cuda_number(&generic.bounds)?));
					}
				},
				GenericParam::Lifetime(generic) => {
					return Err(syn::Error::new_spanned(generic, "lifetime generic parameters not supported"));
				},
				GenericParam::Const(generic) => {
					return Err(syn::Error::new_spanned(generic, "const generic parameters not supported"));
				}
			}
		}
		
		// Process where bounds
		if let Some(whre) = &f.decl.generics.where_clause {
			for predicate in &whre.predicates {
				match predicate {
					WherePredicate::Type(predicate) => {
						if let Some(lts) = &predicate.lifetimes {
							return Err(syn::Error::new_spanned(lts, "lifetimes on generic predicates not supported"));
						}
						if !type_param_bound_is_cuda_number(&predicate.bounds)? {
							return Err(syn::Error::new_spanned(&predicate.bounds, "generic param must be bound by `CudaNumber`, e.g. `T: CudaNumber`"));
						}
						let path = get_type_path(&predicate.bounded_ty)?;
						let mut found = false;
						for (gen_ident, _, has_cuda_number_bound) in generics.iter_mut() {
							if path.is_ident(gen_ident.clone()) {
								found = true;
								*has_cuda_number_bound = true;
								break;
							}
						}
						if !found {
							let mut generic_idents = vec![];
							for (gen_ident, _, _) in generics.iter() {
								generic_idents.push(gen_ident.to_string());
							}
							return Err(syn::Error::new_spanned(path, format!("unknown type path (must be one of {:?})", generic_idents)));
						}
					},
					WherePredicate::Lifetime(predicate) => {
						return Err(syn::Error::new_spanned(predicate, "lifetime generic predicates not supported"));
					},
					WherePredicate::Eq(predicate) => {
						return Err(syn::Error::new_spanned(predicate, "equal generic predicates not supported"));
					},
				}
			}
		}
		
		// Check every generic has a cuda number bound
		for (_, span, has_cuda_number_bound) in generics.iter() {
			if !has_cuda_number_bound {
				return Err(syn::Error::new(span.clone(), "generic param must be bound by `CudaNumber`, e.g. `T: CudaNumber`"));
			}
		}
		
		let number_generics: Vec<String> = generics.iter().map(|(i, _, _)| i.to_string()).collect();
		let type_args = number_generics.iter().map(|ty| format!("typename {}", ty)).collect::<Vec<String>>().join(", ");
		let c_template_decl = format!("template<{}>", type_args);
		
		Ok(FnInfo {
			ty: fn_type,
			number_generics,
			c_template_decl,
		})
	}

	pub fn is_generic(&self) -> bool {
		self.number_generics.len() > 0
	}
}

/// Try to get a type path from a type. If the type is not a type path, then return an error.
pub fn get_type_path(ty: &syn::Type) -> Result<&syn::Path, syn::Error> {
	use syn::Type;
	match ty {
		Type::Path(p) => {
			if let Some(_) = &p.qself {
				Err(syn::Error::new_spanned(p, "qself <_>:: type path not supported"))
			} else {
				Ok(&p.path)
			}
		},
		Type::Paren(ty) => {
			get_type_path(&ty.elem)
		},
		Type::Group(ty) => {
			get_type_path(&ty.elem)
		},
		_ => {
			Err(syn::Error::new_spanned(ty, "type is not a type path"))
		}
	}
}

pub fn tokens_join2(t1: impl ToTokens, t2: impl ToTokens) -> TokenStream {
	let mut t = TokenStream::new();
	t1.to_tokens(&mut t);
	t2.to_tokens(&mut t);
	t
}

pub fn tokens_join<I, T>(ts: I) -> TokenStream where I: IntoIterator<Item=T>, T: ToTokens {
	let mut t = TokenStream::new();
	for ts in ts {
		ts.to_tokens(&mut t);
	}
	t
}

/// Checks if an identifier has the value given
pub fn ident_eq(ident: &syn::Ident, value: &str) -> bool {
	ident.to_string() == value
}

/// Checks a type path and a string that represents a type path, to see if they match
pub fn type_path_matches(path: &syn::Path, mut ty: &str) -> bool {
	if ty.starts_with("::") {
		ty = &ty["::".len()..];
	}
	let segment_count = ty.split("::").count();
	if path.leading_colon.is_some() {
		// Have to match fully
		if segment_count != path.segments.len() {
			return false;
		}
		for (i, segment) in ty.split("::").enumerate() {
			if i >= path.segments.len() {
				break;
			}
			if !ident_eq(&path.segments[i].ident, segment)
					|| !path.segments[i].arguments.is_empty() {
				return false;
			}
		}
		true
	} else {
		// Can match in any place
		let mut ty_it = ty.rsplit("::");
		for i in (0..path.segments.len()).rev() {
			let segment = match ty_it.next() {
				Some(s) => s,
				None => return false,
			};
			// println!("{}, {}: {}", i, &segment, &path.segments[i].ident);
			if !ident_eq(&path.segments[i].ident, segment)
					|| !path.segments[i].arguments.is_empty() {
				return false;
			}
		}
		true
	}
}


#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum FunctionType {
	Host,
	Device,
	Global,
}
impl FunctionType {
	pub fn try_from_attr(attr: &syn::Attribute) -> Option<Self> {
		Self::try_from_path(&attr.path)
	}

	pub fn try_from_path(path: &syn::Path) -> Option<Self> {
		use crate::FunctionType::*;

		for &ty in [Host, Device, Global].iter() {
			if crate::type_path_matches(&path, &format!("::cuda_macros_impl::{}", ty.attr())) {
				return Some(ty);
			} else if crate::type_path_matches(&path, &format!("::cuda_macros::{}", ty.attr())) {
				return Some(ty);
			} else if crate::type_path_matches(&path, &format!("::cuda_util::{}", ty.attr())) {
				return Some(ty);
			} else if crate::type_path_matches(&path, &format!("::cuda::{}", ty.attr())) {
				return Some(ty);
			}
		}
		None
	}

	pub fn attr(self: &Self) -> &'static str {
		match self {
			FunctionType::Host => "host",
			FunctionType::Device => "device",
			FunctionType::Global => "global",
		}
	}

	pub fn cattr(self: &Self) -> &'static str {
		match self {
			FunctionType::Host => "__host__",
			FunctionType::Device => "__device__",
			FunctionType::Global => "__global__",
		}
	}
}


#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_ident_eq() {
		assert!(ident_eq(&syn::parse_str::<syn::Ident>("i32").unwrap(), "i32"));
		assert!(!ident_eq(&syn::parse_str::<syn::Ident>("i32").unwrap(), "f32"));
	}

	#[test]
	fn test_type_path_matches() {
		assert!(type_path_matches(&syn::parse_str("::std::os::raw::c_long").unwrap(), "::std::os::raw::c_long"));
		assert!(type_path_matches(&syn::parse_str("std::os::raw::c_long").unwrap(), "::std::os::raw::c_long"));
		assert!(type_path_matches(&syn::parse_str("os::raw::c_long").unwrap(), "::std::os::raw::c_long"));
		assert!(type_path_matches(&syn::parse_str("raw::c_long").unwrap(), "::std::os::raw::c_long"));
		assert!(type_path_matches(&syn::parse_str("c_long").unwrap(), "::std::os::raw::c_long"));
		assert!(type_path_matches(&syn::parse_str("i32").unwrap(), "i32"));
		
		assert!(!type_path_matches(&syn::parse_str("::std::os::raw::c_long").unwrap(), "c_long"));
		assert!(!type_path_matches(&syn::parse_str("std::os::raw::c_long").unwrap(), "c_long"));
		assert!(!type_path_matches(&syn::parse_str("os::raw::c_long").unwrap(), "c_long"));
		assert!(!type_path_matches(&syn::parse_str("raw::c_long").unwrap(), "c_long"));
		assert!(type_path_matches(&syn::parse_str("c_long").unwrap(), "c_long"));
		
		assert!(!type_path_matches(&syn::parse_str("std::os::raw::c_long").unwrap(), "::std::unix::raw::c_long"));
		assert!(!type_path_matches(&syn::parse_str("something::std::os::raw::c_long").unwrap(), "::std::os::raw::c_long"));
	}
	
	#[test]
	fn test_function_type_try_from_path() {
		use crate::FunctionType::*;

		assert_eq!(Some(Host)  , FunctionType::try_from_path(&syn::parse_str("host"  ).unwrap()));
		assert_eq!(Some(Device), FunctionType::try_from_path(&syn::parse_str("device").unwrap()));
		assert_eq!(Some(Global), FunctionType::try_from_path(&syn::parse_str("global").unwrap()));

		assert_eq!(Some(Host), FunctionType::try_from_path(&syn::parse_str("::cuda_macros::host").unwrap()));
		assert_eq!(Some(Host), FunctionType::try_from_path(&syn::parse_str("cuda_macros::host").unwrap()));
		assert_eq!(Some(Host), FunctionType::try_from_path(&syn::parse_str("::cuda_macros_impl::host").unwrap()));
		assert_eq!(Some(Host), FunctionType::try_from_path(&syn::parse_str("cuda_macros_impl::host").unwrap()));
		assert_eq!(Some(Host), FunctionType::try_from_path(&syn::parse_str("::cuda_util::host").unwrap()));
		assert_eq!(Some(Host), FunctionType::try_from_path(&syn::parse_str("cuda_util::host").unwrap()));
	}
}
