
use cfg_if::cfg_if;

/// Normal users of this crate should read this as a `Copy` type bound.
/// 
/// However, if the `unstable` feature is enabled, then this trait has no bound.
/// The main type that uses this is the [`GpuVec`](struct.GpuVec.html), and through
/// it the [`GpuSlice`](struct.GpuSlice.html) type.
pub trait CopyIfStable {
	private_decl!();
}
cfg_if! {
	if #[cfg(feature="unstable")] {
		impl<T> CopyIfStable for T {
			private_impl!();
		}
	} else {
		impl<T> CopyIfStable for T where T: Copy {
			private_impl!();
		}
	}
}

/* This doesn't work. Keeping as a reminder.
cfg_if! {
	if #[cfg(feature="unstable")] {
		macro_rules! copy_bound_if_stable {
			($t:ident) => {
				$ident
			};
			($t:ident: $rest:tt) => {
				$ident: $rest
			};
		}
	} else {
		macro_rules! copy_bound_if_stable {
			($t:ident) => {
				$ident: Copy
			};
			($t:ident: $rest:tt) => {
				$ident: Copy + $rest
			};
		}
	}
}
*/

cfg_if! {
	if #[cfg(feature="unstable")] {
		mod has_copy {
			pub trait HasCopy {
				fn has_copy() -> bool;
			}
			impl<T> HasCopy for T {
				default fn has_copy() -> bool {
					false
				}
			}
			impl<T> HasCopy for T where T: Copy {
				fn has_copy() -> bool {
					true
				}
			}
		}
		/// Returns `true` if the bound `T: Copy` is satisfied, or if the `unstable` feature is not enabled.
		pub fn has_copy<T: CopyIfStable>() -> bool {
			<T as has_copy::HasCopy>::has_copy()
		}
	} else {
		/// Returns `true` if the bound `T: Copy` is satisfied, or if the `unstable` feature is not enabled.
		pub fn has_copy<T: CopyIfStable>() -> bool {
			true
		}
	}
}

cfg_if! {
	if #[cfg(feature="unstable")] {
		mod has_drop {
			pub trait HasDrop {
				fn has_drop() -> bool;
			}
			impl<T> HasDrop for T {
				default fn has_drop() -> bool {
					false
				}
			}
			impl<T> HasDrop for T where T: Drop {
				fn has_drop() -> bool {
					true
				}
			}
		}
		/// Returns `true` if the bound `T: Drop` is satisfied, and if the `unstable` feature is not enabled.
		pub fn has_drop<T: CopyIfStable>() -> bool {
			<T as has_drop::HasDrop>::has_drop()
		}
	} else {
		/// Returns `true` if the bound `T: Drop` is satisfied, and if the `unstable` feature is not enabled.
		pub fn has_drop<T: CopyIfStable>() -> bool {
			false
		}
	}
}
