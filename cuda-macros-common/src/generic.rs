
use syn::punctuated::Punctuated;

use super::GpuTypeEnum;


pub fn instantiate_generic_type(ty: &syn::Type, tys: &[(&str, GpuTypeEnum)]) -> syn::Type {
	use syn::Type;

	let mut ty = ty.clone();
	match &mut ty {
		Type::Slice(ty) => {
			*ty.elem.as_mut() = instantiate_generic_type(&ty.elem, tys);
		},
		Type::Array(ty) => {
			*ty.elem.as_mut() = instantiate_generic_type(&ty.elem, tys);
		},
		Type::Ptr(ty) => {
			*ty.elem.as_mut() = instantiate_generic_type(&ty.elem, tys);
		},
		Type::Reference(ty) => {
			*ty.elem.as_mut() = instantiate_generic_type(&ty.elem, tys);
		},
		Type::BareFn(ty) => {
			for arg in ty.inputs.iter_mut() {
				arg.ty = instantiate_generic_type(&arg.ty, tys);
			}
			if let syn::ReturnType::Type(_, ty) = &mut ty.output {
				*ty.as_mut() = instantiate_generic_type(&ty, tys);
			}
		},
		Type::Never(_) => {},
		Type::Tuple(ty) => {
			for arg in ty.elems.iter_mut() {
				*arg = instantiate_generic_type(&arg, tys);
			}
		},
		Type::Path(ty) => {
			if ty.qself.is_none() && ty.path.leading_colon.is_none() && ty.path.segments.len() == 1 {
				for (generic_name, generic_type) in tys.iter() {
					if ty.path.is_ident(generic_name) {
						ty.path.segments[0].ident = syn::Ident::new(generic_type.rust_name(), ty.path.segments[0].ident.span())
					}
				}
			}
		},
		Type::TraitObject(_) => {
			unimplemented!("instantiate_generic_type: Type::TraitObject");
		},
		Type::ImplTrait(_) => {
			unimplemented!("instantiate_generic_type: Type::ImplTrait");
		},
		Type::Paren(ty) => {
			*ty.elem.as_mut() = instantiate_generic_type(&ty.elem, tys);
		},
		Type::Group(ty) => {
			*ty.elem.as_mut() = instantiate_generic_type(&ty.elem, tys);
		},
		Type::Infer(_) => {},
		Type::Macro(_) => {},
		Type::Verbatim(_) => {},
	}
	ty
}

pub fn instantiate_generic_arg(arg: &syn::FnArg, tys: &[(&str, GpuTypeEnum)]) -> Option<syn::FnArg> {
	use syn::FnArg;

	match arg {
		FnArg::SelfRef(_) | FnArg::SelfValue(_) | FnArg::Inferred(_) => Some(arg.clone()),
		FnArg::Captured(arg) => {
			Some(FnArg::Captured(syn::ArgCaptured {
				pat: arg.pat.clone(),
				colon_token: arg.colon_token.clone(),
				ty: instantiate_generic_type(&arg.ty, tys)
			}))
		},
		FnArg::Ignored(__global__) => {
			//FnArg::Ignored(instantiate_generic_type(ty, tys))
			None
		}
	}
}

pub fn instantiate_generic_args(args: &Punctuated<syn::FnArg, syn::token::Comma>, tys: &[(&str, GpuTypeEnum)]) -> Punctuated<syn::FnArg, syn::token::Comma> {
	use syn::punctuated::Pair;

	args.pairs().filter_map(|pair| match pair {
		Pair::Punctuated(arg, p) => {
			instantiate_generic_arg(arg, tys).map(|x| Pair::Punctuated(x, p.clone()))
		},
		Pair::End(arg) => {
			instantiate_generic_arg(arg, tys).map(|x| Pair::End(x))
		}
	}).collect()
}
