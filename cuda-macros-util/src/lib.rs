
extern crate cuda;
extern crate proc_macro2;
extern crate syn;
extern crate quote;


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

		for (name, &ty) in ["host", "device", "global"].iter().zip([Host, Device, Global].iter()) {
			if crate::type_path_matches(&path, &format!("::cuda_macros_impl::{}", name)) {
				return Some(ty);
			} else if crate::type_path_matches(&path, &format!("::cuda_macros::{}", name)) {
				return Some(ty);
			}
		}
		None
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
	}
}