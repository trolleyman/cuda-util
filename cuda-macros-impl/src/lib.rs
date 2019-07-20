
use proc_macro2::TokenStream;
use syn::parse_macro_input;
use quote::{quote, ToTokens};
use quote::TokenStreamExt;

use cuda_macros_util::FunctionType


fn process_host_fn(f: syn::ItemFn) -> TokenStream {
    // Pass through
    // TODO: handle cfg(__CUDA_ARCH__) / cfg(__CUDACC__)
    // TODO: https://stackoverflow.com/questions/23899190/can-a-host-device-function-know-where-it-is-executing
    quote!{ #f }
}

fn process_device_fn(_f: syn::ItemFn) -> TokenStream {
    // Output nothing - already compiled device version in build.rs
}

fn process_global_fn(f: syn::ItemFn) -> TokenStream {
    // Create function wrapper & link to CUDA function
    let ident = f.ident.clone();
    let c_ident = quote!{ _rust_cuda_#ident };
    // TODO
    let visibility = unimplemented!();
    let content = unimplemented!();
    let args = unimplemented!();
    quote!{
        extern "C" fn #c_ident;

        unsafe #visibility fn #ident(config: impl ::std::convert::Into<::cuda_macros::ExecutionConfig>, #args) {
            #c_ident()
        }
    }
}

fn process_fn(mut f: syn::ItemFn, fn_type: FunctionType) -> TokenStream {
    use cuda_macros_util::FunctionType::*;

    let mut device_host = false;
    for a in f.attrs.iter() {
        if let Some(t) = FunctionType::try_from_attr(a) {
            match (fn_type, t) {
                (Host, Device) | (Device, Host) => device_host = true,
                _ => return syn::Error::new_spanned(a, "invalid combination of CUDA function attributes")
                        .to_compile_error().into();
            }
        } else {
            // Invalid attribute
            return syn::Error::new_spanned(a, "no other attributes allowed on CUDA functions")
                .to_compile_error().into();
        }
    }
    // Strip function attributes
    f.attrs = vec![];

    // Process functions with both #[device] & #[host]
    if device_host && fn_type == Host {
        if fn_type == Host {
            process_device_fn(f.clone())
        } else if fn_type == Device {
            process_host_fn(f.clone())
        }
    }

    match fn_type {
        Host => process_host_fn(f),
        Device => process_device_fn(f),
        Global => process_global_fn(f),
    }
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
