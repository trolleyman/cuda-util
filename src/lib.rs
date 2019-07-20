
extern crate proc_macro;
extern crate proc_macro2;
extern crate quote;
extern crate syn;

use proc_macro2::TokenStream;
use syn::parse_macro_input;
use quote::{quote, ToTokens};
use quote::TokenStreamExt;


#[derive(Copy, Clone)]
enum FunctionType {
    Host,
    Device,
    Global,
}

fn process_host_fn(f: syn::ItemFn) -> TokenStream {
    // TODO
    for a in f.attrs.iter() {
        println!("{}", a.clone().into_token_stream());
    }
    dbg!(&f.ident);
    println!();
    quote!{ #f }
}

fn process_device_fn(f: syn::ItemFn) -> TokenStream {
    // TODO
    for a in f.attrs.iter() {
        println!("{}", a.clone().into_token_stream());
    }
    dbg!(&f.ident);
    println!();
    quote!{ #f }
}

fn process_global_fn(f: syn::ItemFn) -> TokenStream {
    // TODO
    for a in f.attrs.iter() {
        println!("{}", a.clone().into_token_stream());
    }
    dbg!(&f.ident);
    println!();
    quote!{ #f }
}

fn process_fn(f: syn::ItemFn, fn_type: FunctionType) -> TokenStream {
    match fn_type {
        FunctionType::Host => process_host_fn(f),
        FunctionType::Device => process_device_fn(f),
        FunctionType::Global => process_global_fn(f),
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
