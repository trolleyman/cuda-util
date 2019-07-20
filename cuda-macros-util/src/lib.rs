

/// Checks if an identifier has the value given
pub fn ident_eq(ident: &syn::Ident, value: &str) -> bool {
    ident.to_string() == value
}


#[derive(Copy, Clone)]
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
                    && ident_eq(attr.path.segments[0].ident, "cuda-macros-impl") {
                parse_ident(&attr.path.segments[1].ident)
            } else {
                None
            }
        } else if attr.path.segments.len() == 1 {
            parse_ident(&attr.path.segments[0].ident)
        } else if attr.path.segments.len() == 2
                && ident_eq(attr.path.segments[0].ident, "cuda-macros-impl") {
            parse_ident(&attr.path.segments[1].ident)
        } else {
            None
        }
    }
}
