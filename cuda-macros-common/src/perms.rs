
fn permutations_err_rec<E, T: Clone, F>(values: &[T], len: usize, v: &mut Vec<T>, f: &mut F) -> Result<(), E> where F: FnMut(&[T]) -> Result<(), E> {
	if len == 0 {
		// Call `f`
		f(&v)
	} else {
		for val in values.iter() {
			v.push(val.clone());
			if let Err(e) = permutations_err_rec(values, len-1, v, f) {
				return Err(e);
			}
			v.pop();
		}
		Ok(())
	}
}

/// Get all permutations of `values` with length `len`, and calls a given function with each result. Stops early if an error `E` is encountered, and returns `Err`.
pub fn permutations_err<E, T: Clone, F>(values: &[T], len: usize, mut f: F) -> Result<(), E> where F: FnMut(&[T]) -> Result<(), E> {
	let mut v = Vec::with_capacity(len);
	permutations_err_rec(values, len, &mut v, &mut f)
}

#[cfg(test)]
#[test]
fn test_permutations_err() {
	let values = &[1, 2];
	let mut v = Vec::new();
	permutations_err(values, 3, |arr| { v.push(Vec::from(arr)); if arr == &[2, 1, 2] { Err(()) } else { Ok(()) } });
	let expected = &[
		&[1, 1, 1],
		&[1, 1, 2],
		&[1, 2, 1],
		&[1, 2, 2],
		&[2, 1, 1],
		&[2, 1, 2],
		// Early stop test
	];
	assert_eq!(v, expected);
}

/// Get all permutations of `values` with length `len`, and calls a given function with each result.
pub fn permutations<F, T: Clone>(values: &[T], len: usize, mut f: F) where F: FnMut(&[T]) {
	permutations_err::<(), _, _>(values, len, |perm| { f(perm); Ok(()) }).unwrap()
}

#[cfg(test)]
#[test]
fn test_permutations() {
	let values = &[1, 2];
	let mut v = Vec::new();
	permutations(values, 3, |arr| v.push(Vec::from(arr)));
	let expected = &[
		&[1, 1, 1],
		&[1, 1, 2],
		&[1, 2, 1],
		&[1, 2, 2],
		&[2, 1, 1],
		&[2, 1, 2],
		&[2, 2, 1],
		&[2, 2, 2],
	];
	assert_eq!(v, expected);
}
