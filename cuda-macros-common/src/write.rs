//! Functions used for outputting Rust objects to CUDA source and header files

use std::io::prelude::*;
use std::io::{self, SeekFrom};
use std::borrow::Cow;
use std::fmt;

use chrono::Local;

use super::{FunctionType, FnInfo, GpuTypeEnum, permutations_err, conv};
use super::file::*;


#[derive(Debug)]
pub enum TransError {
	IoError(io::Error),
	SynError(syn::Error),
}
impl From<io::Error> for TransError {
	fn from(e: io::Error) -> Self {
		TransError::IoError(e)
	}
}
impl From<syn::Error> for TransError {
	fn from(e: syn::Error) -> Self {
		TransError::SynError(e)
	}
}
impl fmt::Display for TransError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			TransError::IoError(e) => write!(f, "io error: {}", e),
			TransError::SynError(e) => write!(f, "syntax error: {}", e),
		}
	}
}
impl std::error::Error for TransError {
	fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
		match self {
			TransError::IoError(e) => Some(e),
			TransError::SynError(e) => Some(e),
		}
	}
}

/// Check if  afunction signature is valid
fn check_function_signature(f: &syn::ItemFn) -> Result<(), syn::Error> {
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
	// if f.decl.generics.params.len() != 0 {
	// 	return Err(syn::Error::new_spanned(f.decl.generics.params.clone(), "generic CUDA functions are not allowed"));
	// }
	// if let Some(item) = &f.decl.generics.where_clause {
	// 	return Err(syn::Error::new_spanned(item.clone(), "generic CUDA functions are not allowed"));
	// }
	Ok(())
}

/// Write function to `header_file` and `src_file`
pub fn write_fn<F: FileLike, G: FileLike>(header_file: &mut F, src_file: &mut G, f: &syn::ItemFn, fn_info: &FnInfo) -> Result<(), TransError> {
	check_function_signature(f)?;
	write_fn_header_file(header_file, f, fn_info)?;
	write_fn_source_file(src_file, f, fn_info)?;
	Ok(())
}

/// Write start of header file
fn write_header_intro(of: &mut dyn FileLike) -> Result<(), TransError> {
	writeln!(of, "/* Generated by cuda-macros https://github.com/trolleyman/cuda-macros on {} */", Local::now().to_rfc2822())?;
	writeln!(of, "{}", r#"#pragma once

#include <stdint.h>
#include <stdbool.h>

#include <cuda_runtime.h>

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

typedef struct {
	uint32_t x;
	uint32_t y;
	uint32_t z;
} rust_cuda_macros_dim3_t;

typedef struct {
	rust_cuda_macros_dim3_t grid_size;
	rust_cuda_macros_dim3_t block_size;
	size_t shared_mem_size;
	cudaStream_t cuda_stream;
} rust_cuda_macros_execution_config_t;

"#)?;
	Ok(())
}

/// Write header file source code to file
fn write_fn_header_file<F: FileLike>(of: &mut F, f: &syn::ItemFn, fn_info: &FnInfo) -> Result<(), TransError> {
	// Check if file hasn't got header
	if of.seek(SeekFrom::End(0))? == 0 {
		write_header_intro(of)?;
	}

	// Write function
	if fn_info.ty == FunctionType::Global {
		if fn_info.is_generic() {
			permutations_err::<TransError, _, _>(GpuTypeEnum::types(), fn_info.number_generics.len(), |tys| {
				write_fn_decl(of, f, fn_info, true, Some(tys))?;
				writeln!(of, ";")?;
				Ok(())
			})?;
		} else {
			write_fn_decl(of, f, fn_info, true, None)?;
			writeln!(of, ";")?;
		}
	}
	write_fn_decl(of, f, fn_info, false, None)?;
	writeln!(of, ";\n")?;
	Ok(())
}

/// Write start of source file
fn write_source_intro(of: &mut dyn FileLike) -> Result<(), TransError> {
	writeln!(of, "/* Generated by cuda-macros https://github.com/trolleyman/cuda-macros on {} */", Local::now().to_rfc2822())?;
	writeln!(of, "")?;
	writeln!(of, "#include <cstdio>")?;
	writeln!(of, "#include <header.h>")?;
	writeln!(of, "")?;
	Ok(())
}

/// Write function to file
fn write_fn_source_file<F: FileLike>(of: &mut F, f: &syn::ItemFn, fn_info: &FnInfo) -> Result<(), TransError> {
	// Check if file hasn't got header
	if of.seek(SeekFrom::End(0))? == 0 {
		write_source_intro(of)?;
	}

	// Write wrapper function
	if fn_info.ty == FunctionType::Global {
		if fn_info.is_generic() {
			permutations_err::<TransError, _, _>(GpuTypeEnum::types(), fn_info.number_generics.len(), |tys| {
				write_fn_decl(of, f, fn_info, true, Some(tys))?;
				writeln!(of, " {{")?;
				write_wrapper_fn_body(of, f, fn_info, Some(tys))?;
				writeln!(of, "}}")?;
				Ok(())
			})?;
		} else {
			write_fn_decl(of, f, fn_info, true, None)?;
			writeln!(of, " {{")?;
			write_wrapper_fn_body(of, f, fn_info, None)?;
			writeln!(of, "}}")?;
		}
	}

	// Write CUDA function
	write_fn_decl(of, f, fn_info, false, None)?;
	writeln!(of, " {{")?;
	write_cuda_fn_body(of, fn_info, &f.block.stmts)?;
	writeln!(of, "}}\n")?;
	Ok(())
}

/// Write function declaration, e.g. `void hello()`
fn write_fn_decl<F: FileLike>(of: &mut F, f: &syn::ItemFn, fn_info: &FnInfo, is_wrapper: bool, generic_tys: Option<&[GpuTypeEnum]>) -> Result<(), TransError> {
	let ret: Cow<'static, str> = match &f.decl.output {
		syn::ReturnType::Default => "void".into(),
		syn::ReturnType::Type(_, ty) => conv::rust_type_to_c(&ty, fn_info, false)
			.map_err(|e| e.unwrap_or_else(|| syn::Error::new_spanned(ty, "invalid return type")))?,
	};
	if fn_info.ty == FunctionType::Global && ret != "void" {
		return Err(syn::Error::new_spanned(f.decl.output.clone(), "invalid return type: #[global] functions must return nothing"))?;
	}
	let mut args = vec![];
	if is_wrapper {
		// Add ExecutionConfig argument
		args.push("rust_cuda_macros_execution_config_t rust_cuda_macros_config".into());
	}
	if let Some(generic_tys) = generic_tys {
		let generic_tys: Vec<_> = fn_info.number_generics.iter().zip(generic_tys.iter()).map(|(x, &y)| (x.as_str(), y)).collect();
		let args_decl = super::instantiate_generic_args(&f.decl.inputs, generic_tys.as_slice());
		for arg in args_decl.iter() {
			if let Some((ty, ident)) = conv::rust_fn_arg_to_c(&arg, fn_info)? {
				args.push(format!("{} {}", ty, ident));
			}
		}
	} else {
		for arg in f.decl.inputs.iter() {
			if let Some((ty, ident)) = conv::rust_fn_arg_to_c(&arg, fn_info)? {
				args.push(format!("{} {}", ty, ident));
			}
		}
	}
	
	let args = args.join(", ");
	let mut fn_ident = if is_wrapper {
		format!("rust_cuda_macros_wrapper_{}_{}", fn_info.number_generics.len(), &f.ident)
	} else {
		format!("{}", &f.ident)
	};
	if let Some(tys) = generic_tys {
		for ty in tys.iter() {
			fn_ident.push('_');
			fn_ident.push_str(ty.rust_name());
		}
	}
	let attr = if is_wrapper { "".into() } else { format!("{} ", fn_info.ty.cattr()) };
	writeln!(of, "/* {} */", &f.ident/* TODO: &f.decl.inputs.iter().map(|arg| arg.to_string()).collect::<String>().join(", ") */)?;
	if fn_info.is_generic() && !is_wrapper {
		writeln!(of, "{}", fn_info.c_template_decl)?;
	}
	write!(of, "EXTERN_C {}{} {}({})", attr, ret, fn_ident, args)?;
	Ok(())
}

/// Wrapper function body: all this does is call the CUDA function
fn write_wrapper_fn_body<F: FileLike>(of: &mut F, f: &syn::ItemFn, fn_info: &FnInfo, generic_tys: Option<&[GpuTypeEnum]>) -> Result<(), TransError> {
	let mut args = vec![];
	for arg in f.decl.inputs.iter() {
		if let Some((_, ident)) = conv::rust_fn_arg_to_c(&arg, fn_info)? {
			args.push(format!("{}", ident));
		}
	}
	let args = args.join(", ");
	
	if let Some(tys) = generic_tys {
		write!(of, "\treturn {}<", &f.ident)?;
		for (i, ty) in tys.iter().enumerate() {
			write!(of, "{}", ty.c_name())?;
			if i != tys.len() - 1 {
				write!(of, ", ")?;
			}
		}
		writeln!(of, "><<<")?;
	} else {
		writeln!(of, "\treturn {}<<<", &f.ident)?;
	}
	writeln!(of, "\t\tdim3(rust_cuda_macros_config.grid_size.x, rust_cuda_macros_config.grid_size.y, rust_cuda_macros_config.grid_size.z),")?;
	writeln!(of, "\t\tdim3(rust_cuda_macros_config.block_size.x, rust_cuda_macros_config.block_size.y, rust_cuda_macros_config.block_size.z),")?;
	writeln!(of, "\t\trust_cuda_macros_config.shared_mem_size,")?;
	writeln!(of, "\t\trust_cuda_macros_config.cuda_stream")?;
	writeln!(of, "\t>>>({});", args)?;
	Ok(())
}

fn write_cuda_fn_body<F: FileLike>(of: &mut F, fn_info: &FnInfo, stmts: &[syn::Stmt]) -> Result<(), TransError> {
	write_stmts(FileLikeIndent::new(of, 1), fn_info, stmts)
}

fn write_stmts<F: FileLike>(mut of: FileLikeIndent<F>, fn_info: &FnInfo, stmts: &[syn::Stmt]) -> Result<(), TransError> {
	for stmt in stmts.iter() {
		write_stmt(of.clone(), fn_info, stmt)?;
	}
	Ok(())
}

fn write_stmt<F: FileLike>(mut of: FileLikeIndent<F>, fn_info: &FnInfo, stmt: &syn::Stmt) -> Result<(), TransError> {
	use syn::{Pat, Stmt};

	match stmt {
		Stmt::Local(local) => {
			if conv::is_item_enabled(&local.attrs, conv::CfgType::DeviceCode)? {
				let mut is_shared = false;
				for attr in &local.attrs {
					if attr.path.is_ident("shared") {
						is_shared = true;
					}
				}
				if local.pats.len() != 1 {
					Err(syn::Error::new_spanned(local.pats.clone(), "invalid pattern: only simple identifier allowed"))?;
				}
				let pat = &local.pats[0];
				let (ident, mutability) = match pat {
					Pat::Ident(pident) if pident.by_ref == None && pident.subpat == None => {
						(pident.ident.to_string(), pident.mutability.is_some())
					},
					_ => return Err(syn::Error::new_spanned(local.pats.clone(), "invalid pattern: only simple identifier allowed").into()),
				};
				if local.ty.is_none() {
					return Err(syn::Error::new_spanned(local.clone(), "inferred types are not supported").into());
				}
				let ty = &local.ty.as_ref().unwrap();
				let (cty_prefix, cty_postfix) = if is_shared {
					conv::rust_shared_type_to_c(&ty.1, fn_info)?
				} else {
					(conv::rust_type_to_c(&ty.1, fn_info, !mutability)
						.map_err(|e| e.unwrap_or(syn::Error::new_spanned(ty.1.clone(), "invalid type")))?, "".into())
				};
				write!(&mut of, "{} ", cty_prefix)?;
				write!(&mut of, "{}", ident)?;
				write!(&mut of, "{}", cty_postfix)?;
				if let Some((_, init)) = &local.init {
					if is_shared {
						return Err(syn::Error::new_spanned(init.clone(), "initializers not allowed in #[shared] declarations").into());
					}
					write!(&mut of, " = ")?;
					write_expr(of.clone(), fn_info, &init)?;
				}
				writeln!(&mut of, ";")?;
			}
		},
		Stmt::Item(item) => {
			/*
			match item {
				// TODO: syn::Item::Struct => {},
				// TODO: syn::Item::Enum => {},
				// TODO: syn::Item::Union => {},
			}*/
			Err(syn::Error::new_spanned(item.clone(), "item not allowed"))?;
		},
		Stmt::Expr(e) => {
			write_expr(of.clone(), fn_info, &e)?;
		},
		Stmt::Semi(e, _) => {
			write_expr(of.clone(), fn_info, &e)?;
			writeln!(&mut of, ";")?;
		}
	}
	Ok(())
}

fn expr_requires_brackets(e: &syn::Expr) -> bool {
	use syn::Expr;

	match e {
		Expr::Path(_) | Expr::Field(_) | Expr::Lit(_) => false,
		_ => true
	}
}

fn write_expr_with_brackets<F: FileLike>(mut of: FileLikeIndent<F>, fn_info: &FnInfo, e: &syn::Expr) -> Result<(), TransError> {
	let brackets = expr_requires_brackets(e);
	if brackets {
		write!(&mut of, "(")?;
	}
	write_expr(of.clone(), fn_info, e)?;
	if brackets {
		write!(&mut of, ")")?;
	}
	Ok(())
}

fn write_expr<F: FileLike>(mut of: FileLikeIndent<F>, fn_info: &FnInfo, e: &syn::Expr) -> Result<(), TransError> {
	use syn::{UnOp, Expr, Member};

	match e {
		Expr::Box(e) => Err(syn::Error::new_spanned(e.clone(), "box expressions are not supported"))?,
		Expr::InPlace(e) => Err(syn::Error::new_spanned(e.clone(), "placement expressions are not supported"))?,
		Expr::Array(e) => {
			if conv::is_item_enabled(&e.attrs, conv::CfgType::DeviceCode)? {
				write!(&mut of, "[")?;
				for (i, elem) in e.elems.iter().enumerate() {
					write_expr(of.clone(), fn_info, &elem)?;
					if i < e.elems.len() - 1 {
						write!(&mut of, ",")?;
					}
				}
				write!(&mut of, "]")?;
			}
		},
		Expr::Call(e) => {
			if conv::is_item_enabled(&e.attrs, conv::CfgType::DeviceCode)? {
				write_expr_with_brackets(of.clone(), fn_info, &*e.func)?;
				write!(&mut of, "(")?;
				for (i, elem) in e.args.iter().enumerate() {
					write_expr(of.clone(), fn_info, &elem)?;
					if i < e.args.len() - 1 {
						write!(&mut of, ", ")?;
					}
				}
				write!(&mut of, ")")?;
			}
		},
		Expr::MethodCall(e) =>{
			if conv::is_item_enabled(&e.attrs, conv::CfgType::DeviceCode)? {
				if e.turbofish.is_some() {
					Err(syn::Error::new_spanned(e.turbofish.clone(), "generic methods are not supported"))?;
				}
				write_expr_with_brackets(of.clone(), fn_info, &*e.receiver)?;
				write!(&mut of, ",")?;
				write!(&mut of, "{}", &e.method)?;
				write!(&mut of, "(")?;
				for (i, elem) in e.args.iter().enumerate() {
					write_expr(of.clone(), fn_info, &elem)?;
					if i < e.args.len() - 1 {
						write!(&mut of, ", ")?;
					}
				}
				write!(&mut of, ")")?;
			}
		},
		Expr::Tuple(e) => Err(syn::Error::new_spanned(e.clone(), "tuple expressions are not supported"))?,
		Expr::Binary(e) => {
			if conv::is_item_enabled(&e.attrs, conv::CfgType::DeviceCode)? {
				write_expr_with_brackets(of.clone(), fn_info, &*e.left)?;
				write!(&mut of, " ")?;
				write_binop(of.clone(), &e.op)?;
				write!(&mut of, " ")?;
				write_expr_with_brackets(of.clone(), fn_info, &*e.right)?;
			}
		},
		Expr::Unary(e) => {
			if conv::is_item_enabled(&e.attrs, conv::CfgType::DeviceCode)? {
				let op = match &e.op {
					UnOp::Deref(_) => '*',
					UnOp::Not(_) => '!',
					UnOp::Neg(_) => '-',
				};
				write!(&mut of, "{}", op)?;
				write_expr(of.clone(), fn_info, &*e.expr)?;
			}
		},
		Expr::Lit(e) => {
			if conv::is_item_enabled(&e.attrs, conv::CfgType::DeviceCode)? {
				write_lit(of.clone(), &e.lit)?;
			}
		},
		Expr::Cast(_e) => {
			unimplemented!("Expr::Cast"); // TODO
		},
		Expr::Type(e) => Err(syn::Error::new_spanned(e.clone(), "type ascription is not supported"))?,
		Expr::Let(e) => Err(syn::Error::new_spanned(e.clone(), "let guards are not supported"))?,
		Expr::If(e) => {
			if conv::is_item_enabled(&e.attrs, conv::CfgType::DeviceCode)? {
				write!(&mut of, "if (")?;
				write_expr(of.incr(), fn_info, &e.cond)?;
				writeln!(&mut of, ") {{")?;
				write_stmts(of.incr(), fn_info, &e.then_branch.stmts)?;
				if let Some((_, else_branch)) = &e.else_branch {
					writeln!(&mut of, "}} else {{")?;
					write_expr(of.incr(), fn_info, &else_branch)?;
				}
				writeln!(&mut of, "}}")?;
			}
		},
		Expr::While(_e) => {
			unimplemented!("Expr::While"); // TODO
		},
		Expr::ForLoop(e) => {
			if conv::is_item_enabled(&e.attrs, conv::CfgType::DeviceCode)? {
				if let Some(label) = &e.label {
					return Err(syn::Error::new_spanned(label.clone(), "labels on for loops are not supported").into());
				}
				
				// Get identifier
				let ident: Cow<'static, str> = match &*e.pat {
					syn::Pat::Wild(_) => "_wild".into(),
					syn::Pat::Ident(pat) => {
						if let Some(r) = &pat.by_ref {
							return Err(syn::Error::new_spanned(r.clone(), "`ref` bindings are not allowed").into());
						}
						if let Some(m) = &pat.mutability {
							return Err(syn::Error::new_spanned(m.clone(), "`mut` bindings are not allowed").into());
						}
						if let Some((_, pat)) = &pat.subpat {
							return Err(syn::Error::new_spanned(pat.clone(), "subpatterns are not allowed").into());
						}
						pat.ident.to_string().into()
					},
					_ => return Err(syn::Error::new_spanned(e.pat.clone(), "only ident patterns are allowed (e.g. `i`)").into()),
				};
				
				// Get range
				let ((from, to), limits) = match conv::get_inner_expr(&e.expr)? {
					Expr::Range(e) => (match (&e.from, &e.to) {
						(None, None) => return Err(syn::Error::new_spanned(e.clone(), "unbounded ranges are not allowed").into()),
						(Some(_), None) | (None, Some(_)) => return Err(syn::Error::new_spanned(e.clone(), "ranges must be fully defined").into()),
						(Some(from), Some(to)) => (&*from, &*to),
					}, e.limits),
					//Expr::Array(e) => /* TODO */
					_ => return Err(syn::Error::new_spanned(e.clone(), "only range expressions are allowed (e.g. `0..10`)").into()),
				};
				let cmp = match limits {
					syn::RangeLimits::HalfOpen(_) => "<",
					syn::RangeLimits::Closed(_) => "<=",
				};
				
				write!(&mut of, "for (int {} = ", &ident)?;
				write_expr_with_brackets(of.incr().incr(), fn_info, from)?;
				write!(&mut of, "; {} {} ", &ident, cmp)?;
				write_expr_with_brackets(of.incr().incr(), fn_info, to)?;
				writeln!(&mut of, "; {}++) {{", &ident)?;
				write_stmts(of.incr(), fn_info, &e.body.stmts)?;
				writeln!(&mut of, "}}")?;
			}
		},
		Expr::Loop(_e) => {
			unimplemented!("Expr::Loop"); // TODO
		},
		Expr::Match(e) => {
			// TODO
			return Err(syn::Error::new_spanned(e.clone(), "match expressions are not yet supported").into())
		},
		Expr::Closure(e) => Err(syn::Error::new_spanned(e.clone(), "closures are not supported"))?,
		Expr::Unsafe(e) => Err(syn::Error::new_spanned(e.clone(), "unsafe blocks are not supported"))?,
		Expr::Block(e) => {
			if conv::is_item_enabled(&e.attrs, conv::CfgType::DeviceCode)? {
				writeln!(&mut of, "{{")?;
				write_stmts(of.incr(), fn_info, &e.block.stmts)?;
				writeln!(&mut of, "}}")?;
			}
		},
		Expr::Assign(e) => {
			if conv::is_item_enabled(&e.attrs, conv::CfgType::DeviceCode)? {
				write_expr(of.clone(), fn_info, &e.left)?;
				write!(&mut of, " = ")?;
				write_expr(of.clone(), fn_info, &e.right)?;
			}
		},
		Expr::AssignOp(e) => {
			if conv::is_item_enabled(&e.attrs, conv::CfgType::DeviceCode)? {
				write_expr(of.clone(), fn_info, &e.left)?;
				write!(&mut of, " ")?;
				write_binop(of.clone(), &e.op)?;
				write!(&mut of, " ")?;
				write_expr(of.clone(), fn_info, &e.right)?;
			}
		},
		Expr::Field(e) => {
			if conv::is_item_enabled(&e.attrs, conv::CfgType::DeviceCode)? {
				write_expr_with_brackets(of.clone(), fn_info, &e.base)?;
				write!(&mut of, ".")?;
				match &e.member {
					Member::Unnamed(i) => Err(syn::Error::new_spanned(i.clone(), "unnamed field accesses are not supported"))?,
					Member::Named(name) => write!(&mut of, "{}", name)?,
				}
			}
		},
		Expr::Index(e) => {
			if conv::is_item_enabled(&e.attrs, conv::CfgType::DeviceCode)? {
				write_expr_with_brackets(of.clone(), fn_info, &e.expr)?;
				write!(&mut of, "[")?;
				write_expr(of.clone(), fn_info, &e.index)?;
				write!(&mut of, "]")?;
			}
		},
		Expr::Range(e) => Err(syn::Error::new_spanned(e.clone(), "range expressions are not supported"))?,
		Expr::Path(e) => {
			if conv::is_item_enabled(&e.attrs, conv::CfgType::DeviceCode)? {
				if e.qself.is_some() || e.path.leading_colon.is_some() || e.path.segments.len() != 1 {
					Err(syn::Error::new_spanned(e.clone(), "paths are not supported"))?;
				}
				let segment = &e.path.segments[0];
				if segment.arguments != syn::PathArguments::None {
					Err(syn::Error::new_spanned(e.clone(), "paths with generic arguments are not supported"))?;
				}
				write!(&mut of, "{}", &segment.ident)?;
			}
		},
		Expr::Reference(_e) => {
			unimplemented!("Expr::Reference"); // TODO
		},
		Expr::Break(_e) => {
			unimplemented!("Expr::Break"); // TODO
		},
		Expr::Continue(_e) => {
			unimplemented!("Expr::Continue"); // TODO
		},
		Expr::Return(e) => {
			if conv::is_item_enabled(&e.attrs, conv::CfgType::DeviceCode)? {
				if let Some(expr) = &e.expr {
					write!(&mut of, "return ")?;
					write_expr(of.clone(), fn_info, expr)?;
				} else {
					write!(&mut of, "return")?;
				}
			}
		},
		Expr::Macro(e) => Err(syn::Error::new_spanned(e.clone(), "macros are not supported"))?,
		Expr::Struct(_e) => {
			unimplemented!("Expr::Struct"); // TODO
		},
		Expr::Repeat(_e) => {
			unimplemented!("Expr::Repeat"); // TODO
		},
		Expr::Paren(syn::ExprParen{attrs, expr, ..}) | Expr::Group(syn::ExprGroup{attrs, expr, ..}) => {
			if conv::is_item_enabled(attrs, conv::CfgType::DeviceCode)? {
				write!(&mut of, "(")?;
				write_expr(of.clone(), fn_info, expr)?;
				write!(&mut of, ")")?;
			}
		},
		Expr::Try(e) => Err(syn::Error::new_spanned(e.clone(), "try expressions are not supported"))?,
		Expr::Async(e) => Err(syn::Error::new_spanned(e.clone(), "async is not supported"))?,
		Expr::TryBlock(e) => Err(syn::Error::new_spanned(e.clone(), "try blocks are not supported"))?,
		Expr::Yield(e) => Err(syn::Error::new_spanned(e.clone(), "yield is not supported"))?,
		Expr::Verbatim(e) => Err(syn::Error::new_spanned(e.clone(), "unknown expression"))?,
	}
	Ok(())
}

fn write_binop<F: FileLike>(mut of: FileLikeIndent<F>, op: &syn::BinOp) -> Result<(), TransError> {
	use syn::BinOp::*;
	let op = match op {
		Add(_) => "+",
		Sub(_) => "-",
		Mul(_) => "*",
		Div(_) => "/",
		Rem(_) => "%",
		And(_) => "&&",
		Or(_) => "||",
		BitXor(_) => "^",
		BitAnd(_) => "&",
		BitOr(_) => "|",
		Shl(_) => "<<",
		Shr(_) => ">>",
		Eq(_) => "==",
		Lt(_) => "<",
		Le(_) => "<=",
		Ne(_) => "!=",
		Ge(_) => ">=",
		Gt(_) => ">",
		AddEq(_) => "+=",
		SubEq(_) => "-=",
		MulEq(_) => "*=",
		DivEq(_) => "/=",
		RemEq(_) => "%=",
		BitXorEq(_) => "^=",
		BitAndEq(_) => "&=",
		BitOrEq(_) => "|=",
		ShlEq(_) => "<<=",
		ShrEq(_) => ">>=",
	};
	write!(&mut of, "{}", op)?;
	Ok(())
}

fn write_lit<F: FileLike>(mut of: FileLikeIndent<F>, lit: &syn::Lit) -> Result<(), TransError> {
	use syn::Lit::*;
	use syn::{IntSuffix, FloatSuffix};
	match lit {
		Str(lit) => write_string_lit(of.clone(), lit.span(), &lit.value())?,
		ByteStr(lit) => write_string_lit(of.clone(), lit.span(), &lit.value().iter().map(|&c| c as char).collect::<String>())?,
		Byte(lit) => write!(&mut of, "((char){}/*{:?}*/)", lit.value(), lit.value() as char)?,
		Char(lit) => write!(&mut of, "((uint32_t){}/*{:?}*/)", lit.value() as u32, lit.value())?,
		Int(lit) => {
			match lit.suffix() {
				IntSuffix::None  => write!(&mut of, "{}", lit.value())?,
				IntSuffix::I8    => write!(&mut of, "((int8_t){})", lit.value())?,
				IntSuffix::U8    => write!(&mut of, "((uint8_t){})", lit.value())?,
				IntSuffix::I16   => write!(&mut of, "((int16_t){})", lit.value())?,
				IntSuffix::U16   => write!(&mut of, "((uint16_t){})", lit.value())?,
				IntSuffix::I32   => write!(&mut of, "((int32_t){})", lit.value())?,
				IntSuffix::U32   => write!(&mut of, "((uint32_t){})", lit.value())?,
				IntSuffix::I64   => write!(&mut of, "((int64_t){})", lit.value())?,
				IntSuffix::U64   => write!(&mut of, "((uint64_t){})", lit.value())?,
				IntSuffix::I128  => Err(syn::Error::new_spanned(lit.clone(), "128-bit integers are not supported"))?,
				IntSuffix::U128  => Err(syn::Error::new_spanned(lit.clone(), "128-bit integers are not supported"))?,
				IntSuffix::Isize => write!(&mut of, "((isize_t){})", lit.value())?,
				IntSuffix::Usize => write!(&mut of, "((usize_t){})", lit.value())?,
			}
		},
		Float(lit) => {
			match lit.suffix() {
				FloatSuffix::None => write!(&mut of, "{}", lit.value())?,
				FloatSuffix::F32  => write!(&mut of, "((float){})", lit.value())?,
				FloatSuffix::F64  => write!(&mut of, "((double){})", lit.value())?,
			}
		},
		Bool(lit) => write!(&mut of, "{:?}", lit.value)?,
		Verbatim(lit) => Err(syn::Error::new_spanned(lit.clone(), "unparseable literal"))?,
	}
	Ok(())
}

fn write_string_lit<F: FileLike>(mut of: FileLikeIndent<F>, span: proc_macro2::Span, s: &str) -> Result<(), TransError> {
	write!(&mut of, "{}", '"')?;
	for c in s.chars() {
		match c {
			//'\''     => write!(&mut of, r#"\'"#)?,
			'"'      => write!(&mut of, r#"\""#)?,
			'\u{3F}' => write!(&mut of, "\\?")?,
			'\\'     => write!(&mut of, r#"\\"#)?,
			'\u{07}' => write!(&mut of, "\\a")?,
			'\u{08}' => write!(&mut of, "\\b")?,
			'\u{0C}' => write!(&mut of, "\\f")?,
			'\n'     => write!(&mut of, "\\n")?,
			'\r'     => write!(&mut of, "\\r")?,
			'\t'     => write!(&mut of, "\\t")?,
			'\u{0B}' => write!(&mut of, "\\v")?,
			'\u{7F}' => write!(&mut of, "\\x7F")?,  // DEL
			'\u{0}'  ..= '\u{1F}'  => write!(&mut of, "\\x{:02X}", c as u32)?,
			'\u{20}' ..= '\u{7E}'  => write!(&mut of, "{}", c)?,
			// TODO: Handle strings correctly -- "abc" => u8"abc", and b"abc" => "abc"
			_ => return Err(syn::Error::new(span, "non-ASCII string are not supported").into()),
		}
	}
	write!(&mut of, "{}", '"')?;
	Ok(())
}
