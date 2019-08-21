
use std::ops;

use super::{CopyIfStable, GpuSlice};


fn assert_range_valid_msg<T: CopyIfStable>(range: &ops::Range<usize>, slice: &GpuSlice<T>) -> Result<(), String> {
	if range.start > range.end {
		Err(format!("slice index starts at {} but ends at {}", range.start, range.end))
	} else if range.start >= slice.len() {
		Err(format!("index {} out of range for slice of length {}", range.start, slice.len()))
	} else if range.end > slice.len() {
		Err(format!("index {} out of range for slice of length {}", range.end, slice.len()))
	} else {
		Ok(())
	}
}

fn assert_range_valid<T: CopyIfStable>(range: &ops::Range<usize>, slice: &GpuSlice<T>) -> Result<(), ()> {
	if range.start > range.end {
		Err(())
	} else if range.start >= slice.len() {
		Err(())
	} else if range.end > slice.len() {
		Err(())
	} else {
		Ok(())
	}
}

/* Waiting on GAT for this to be viable
	/// Helper trait to allow for [`GpuSlice::get()`](struct.GpuSlice.html#method.get) and [`GpuSlice::get_mut()`](struct.GpuSlice.html#method.get_mut)
	/// to take `I: GpuSliceGet<T>`.
	pub trait GpuSliceGet<T> {
		type Output;
		type OutputMut;
		fn get(&self, slice: &GpuSlice<T>) -> Self::Output;
		fn get_mut(&self, slice: &mut GpuSlice<T>) -> Self::OutputMut;
		
		fn try_get(&self, slice: &GpuSlice<T>) -> Self::Output;
		fn try_get_mut(&self, slice: &mut GpuSlice<T>) -> Self::OutputMut;
	}
	impl<'a, T> GpuSliceGet<T> for usize {
		type Output = GpuRef<'a, T>;
		type OutputMut = GpuMutRef<'a, T>;
		fn get(&'_ self, slice: &'a GpuSlice<T>) -> Self::Output {
			unimplemented!()
		}
		fn get_mut(&'_ self, slice: &'a mut GpuSlice<T>) -> Self::OutputMut {
			unimplemented!()
		}
		
		fn try_get(&'_ self, slice: &'a GpuSlice<T>) -> Self::Output {
			unimplemented!()
		}
		fn try_get_mut(&'_ self, slice: &'a mut GpuSlice<T>) -> Self::OutputMut {
			unimplemented!()
		}
	}
	impl<'a, T, I> GpuSliceGet<T> for I where I: GpuSliceRange<T> {
		type Output = &'a GpuSlice<T>;
		type OutputMut = &'a mut GpuSlice<T>;
		fn get(&'_ self, slice: &'a GpuSlice<T>) -> Self::Output {
			unimplemented!()
		}
		fn get_mut(&'_ self, slice: &'a mut GpuSlice<T>) -> Self::OutputMut {
			unimplemented!()
		}
		
		fn try_get(&'_ self, slice: &'a GpuSlice<T>) -> Self::Output {
			unimplemented!()
		}
		fn try_get_mut(&'_ self, slice: &'a mut GpuSlice<T>) -> Self::OutputMut {
			unimplemented!()
		}
	}
*/

/// Helper trait to allow for slices to be taken easily.
pub trait GpuSliceRange<T: CopyIfStable> {
	fn slice<'a>(&'_ self, slice: &'a GpuSlice<T>) -> &'a GpuSlice<T>;
	fn slice_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> &'a mut GpuSlice<T>;

	fn try_slice<'a>(&'_ self, slice: &'a GpuSlice<T>) -> Option<&'a GpuSlice<T>>;
	fn try_slice_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> Option<&'a mut GpuSlice<T>>;
}
impl<T: CopyIfStable> GpuSliceRange<T> for ops::Range<usize> {
	fn slice<'a>(&'_ self, slice: &'a GpuSlice<T>) -> &'a GpuSlice<T> {
		match assert_range_valid_msg(self, slice) {
			Ok(()) => unsafe { GpuSlice::from_raw_parts(slice.as_ptr().add(self.start), self.end - self.start) },
			Err(e) => panic!(e),
		}
	}
	fn slice_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> &'a mut GpuSlice<T> {
		match assert_range_valid_msg(self, slice) {
			Ok(()) => unsafe { GpuSlice::from_raw_parts_mut(slice.as_mut_ptr().add(self.start), self.end - self.start) },
			Err(e) => panic!(e),
		}
	}

	fn try_slice<'a>(&'_ self, slice: &'a GpuSlice<T>) -> Option<&'a GpuSlice<T>> {
		assert_range_valid(self, slice).ok()?;
		Some(unsafe { GpuSlice::from_raw_parts(slice.as_ptr().add(self.start), self.end - self.start) })
	}
	fn try_slice_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> Option<&'a mut GpuSlice<T>> {
		assert_range_valid(self, slice).ok()?;
		Some(unsafe { GpuSlice::from_raw_parts_mut(slice.as_mut_ptr().add(self.start), self.end - self.start) })
	}
}
impl<T: CopyIfStable> GpuSliceRange<T> for ops::RangeFrom<usize> {
	fn slice<'a>(&'_ self, slice: &'a GpuSlice<T>) -> &'a GpuSlice<T> {
		ops::Range{start: self.start, end: slice.len() }.slice(slice)
	}
	fn slice_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> &'a mut GpuSlice<T> {
		ops::Range{start: self.start, end: slice.len() }.slice_mut(slice)
	}

	fn try_slice<'a>(&'_ self, slice: &'a GpuSlice<T>) -> Option<&'a GpuSlice<T>> {
		ops::Range{start: self.start, end: slice.len() }.try_slice(slice)
	}
	fn try_slice_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> Option<&'a mut GpuSlice<T>> {
		ops::Range{start: self.start, end: slice.len() }.try_slice_mut(slice)
	}
}
impl<T: CopyIfStable> GpuSliceRange<T> for ops::RangeTo<usize> {
	fn slice<'a>(&'_ self, slice: &'a GpuSlice<T>) -> &'a GpuSlice<T> {
		ops::Range{start: 0, end: self.end }.slice(slice)
	}
	fn slice_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> &'a mut GpuSlice<T> {
		ops::Range{start: 0, end: self.end }.slice_mut(slice)
	}

	fn try_slice<'a>(&'_ self, slice: &'a GpuSlice<T>) -> Option<&'a GpuSlice<T>> {
		ops::Range{start: 0, end: self.end }.try_slice(slice)
	}
	fn try_slice_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> Option<&'a mut GpuSlice<T>> {
		ops::Range{start: 0, end: self.end }.try_slice_mut(slice)
	}
}
impl<T: CopyIfStable> GpuSliceRange<T> for ops::RangeFull {
	fn slice<'a>(&'_ self, slice: &'a GpuSlice<T>) -> &'a GpuSlice<T> {
		ops::Range{start: 0, end: slice.len() }.slice(slice)
	}
	fn slice_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> &'a mut GpuSlice<T> {
		ops::Range{start: 0, end: slice.len() }.slice_mut(slice)
	}

	fn try_slice<'a>(&'_ self, slice: &'a GpuSlice<T>) -> Option<&'a GpuSlice<T>> {
		ops::Range{start: 0, end: slice.len() }.try_slice(slice)
	}
	fn try_slice_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> Option<&'a mut GpuSlice<T>> {
		ops::Range{start: 0, end: slice.len() }.try_slice_mut(slice)
	}
}
impl<T: CopyIfStable> GpuSliceRange<T> for ops::RangeInclusive<usize> {
	fn slice<'a>(&'_ self, slice: &'a GpuSlice<T>) -> &'a GpuSlice<T> {
		ops::Range{start: *self.start(), end: *self.end() + 1 }.slice(slice)
	}
	fn slice_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> &'a mut GpuSlice<T> {
		ops::Range{start: *self.start(), end: *self.end() + 1 }.slice_mut(slice)
	}

	fn try_slice<'a>(&'_ self, slice: &'a GpuSlice<T>) -> Option<&'a GpuSlice<T>> {
		ops::Range{start: *self.start(), end: *self.end() + 1 }.try_slice(slice)
	}
	fn try_slice_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> Option<&'a mut GpuSlice<T>> {
		ops::Range{start: *self.start(), end: *self.end() + 1 }.try_slice_mut(slice)
	}
}
impl<T: CopyIfStable> GpuSliceRange<T> for ops::RangeToInclusive<usize> {
	fn slice<'a>(&'_ self, slice: &'a GpuSlice<T>) -> &'a GpuSlice<T> {
		ops::Range{start: 0, end: self.end + 1 }.slice(slice)
	}
	fn slice_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> &'a mut GpuSlice<T> {
		ops::Range{start: 0, end: self.end + 1 }.slice_mut(slice)
	}

	fn try_slice<'a>(&'_ self, slice: &'a GpuSlice<T>) -> Option<&'a GpuSlice<T>> {
		ops::Range{start: 0, end: self.end + 1 }.try_slice(slice)
	}
	fn try_slice_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> Option<&'a mut GpuSlice<T>> {
		ops::Range{start: 0, end: self.end + 1 }.try_slice_mut(slice)
	}
}
