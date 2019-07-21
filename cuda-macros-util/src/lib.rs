

/// Checks if an identifier has the value given
pub fn ident_eq(ident: &syn::Ident, value: &str) -> bool {
	ident.to_string() == value
}

/// Checks a type path and a string that represents a type path, to see if they match
pub fn type_path_matches(type_path: &syn::TypePath, mut ty: &str) -> bool {
	if ty.starts_with("::") {
		ty = &ty["::".len()..];
	}
	let segment_count = ty.split("::").count();
	if type_path.path.leading_colon.is_some() {
		// Have to match fully
		if segment_count != type_path.path.segments.len() {
			return false;
		}
		for (i, segment) in ty.split("::").enumerate() {
			if i >= type_path.path.segments.len() {
				break;
			}
			if !ident_eq(&type_path.path.segments[i].ident, segment)
					|| !type_path.path.segments[i].arguments.is_empty() {
				return false;
			}
		}
		true
	} else {
		// Can match in any place
		for (i, segment) in ty.rsplit("::").enumerate() {
			let j: isize = type_path.path.segments.len() as isize - i as isize - 1;
			if j <= 0 {
				break;
			}
			if !ident_eq(&type_path.path.segments[i].ident, segment)
					|| !type_path.path.segments[i].arguments.is_empty() {
				return false;
			}
		}
		true
	}
}


#[derive(Copy, Clone, PartialEq, Eq)]
pub enum FunctionType {
	Host,
	Device,
	Global,
}
impl FunctionType {
	pub fn try_from_attr(attr: &syn::Attribute) -> Option<Self> {
		fn parse_ident(ident: &syn::Ident) -> Option<FunctionType> {
			if ident_eq(ident, "host") {
				Some(FunctionType::Host)
			} else if ident_eq(ident, "device") {
				Some(FunctionType::Device)
			} else if ident_eq(ident, "global") {
				Some(FunctionType::Global)
			} else {
				None
			}
		}
		if attr.path.leading_colon.is_some() {
			// Match path exactly
			if attr.path.segments.len() == 2
					&& ident_eq(&attr.path.segments[0].ident, "cuda-macros-impl") {
				parse_ident(&attr.path.segments[1].ident)
			} else {
				None
			}
		} else if attr.path.segments.len() == 1 {
			parse_ident(&attr.path.segments[0].ident)
		} else if attr.path.segments.len() == 2
				&& ident_eq(&attr.path.segments[0].ident, "cuda-macros-impl") {
			parse_ident(&attr.path.segments[1].ident)
		} else {
			None
		}
	}
}
