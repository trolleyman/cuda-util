
use std::ops;

use super::GpuSlice;

fn assert_range_valid_msg<T>(range: &ops::Range<usize>, slice: &GpuSlice<T>) -> Result<(), String> {
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

fn assert_range_valid<T>(range: &ops::Range<usize>, slice: &GpuSlice<T>) -> Result<(), ()> {
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

/// Helper trait to allow for slices to be taken easily.
pub trait GpuSliceRange<T> {
	fn index<'a>(&'_ self, slice: &'a GpuSlice<T>) -> &'a GpuSlice<T>;
	fn index_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> &'a mut GpuSlice<T>;

	fn slice<'a>(&'_ self, slice: &'a GpuSlice<T>) -> Option<&'a GpuSlice<T>>;
	fn slice_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> Option<&'a mut GpuSlice<T>>;
}
impl<T> GpuSliceRange<T> for ops::Range<usize> {
	fn index<'a>(&'_ self, slice: &'a GpuSlice<T>) -> &'a GpuSlice<T> {
		match assert_range_valid_msg(self, slice) {
			Ok(()) => unsafe { GpuSlice::from_raw_parts(slice.as_ptr().add(self.start), self.end - self.start) },
			Err(e) => panic!(e),
		}
	}
	fn index_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> &'a mut GpuSlice<T> {
		match assert_range_valid_msg(self, slice) {
			Ok(()) => unsafe { GpuSlice::from_raw_parts_mut(slice.as_mut_ptr().add(self.start), self.end - self.start) },
			Err(e) => panic!(e),
		}
	}

	fn slice<'a>(&'_ self, slice: &'a GpuSlice<T>) -> Option<&'a GpuSlice<T>> {
		assert_range_valid(self, slice).ok()?;
		Some(unsafe { GpuSlice::from_raw_parts(slice.as_ptr().add(self.start), self.end - self.start) })
	}
	fn slice_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> Option<&'a mut GpuSlice<T>> {
		assert_range_valid(self, slice).ok()?;
		Some(unsafe { GpuSlice::from_raw_parts_mut(slice.as_mut_ptr().add(self.start), self.end - self.start) })
	}
}
impl<T> GpuSliceRange<T> for ops::RangeFrom<usize> {
	fn index<'a>(&'_ self, slice: &'a GpuSlice<T>) -> &'a GpuSlice<T> {
		ops::Range{start: self.start, end: slice.len() }.index(slice)
	}
	fn index_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> &'a mut GpuSlice<T> {
		ops::Range{start: self.start, end: slice.len() }.index_mut(slice)
	}

	fn slice<'a>(&'_ self, slice: &'a GpuSlice<T>) -> Option<&'a GpuSlice<T>> {
		ops::Range{start: self.start, end: slice.len() }.slice(slice)
	}
	fn slice_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> Option<&'a mut GpuSlice<T>> {
		ops::Range{start: self.start, end: slice.len() }.slice_mut(slice)
	}
}
impl<T> GpuSliceRange<T> for ops::RangeTo<usize> {
	fn index<'a>(&'_ self, slice: &'a GpuSlice<T>) -> &'a GpuSlice<T> {
		ops::Range{start: 0, end: self.end }.index(slice)
	}
	fn index_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> &'a mut GpuSlice<T> {
		ops::Range{start: 0, end: self.end }.index_mut(slice)
	}

	fn slice<'a>(&'_ self, slice: &'a GpuSlice<T>) -> Option<&'a GpuSlice<T>> {
		ops::Range{start: 0, end: self.end }.slice(slice)
	}
	fn slice_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> Option<&'a mut GpuSlice<T>> {
		ops::Range{start: 0, end: self.end }.slice_mut(slice)
	}
}
impl<T> GpuSliceRange<T> for ops::RangeFull {
	fn index<'a>(&'_ self, slice: &'a GpuSlice<T>) -> &'a GpuSlice<T> {
		ops::Range{start: 0, end: slice.len() }.index(slice)
	}
	fn index_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> &'a mut GpuSlice<T> {
		ops::Range{start: 0, end: slice.len() }.index_mut(slice)
	}

	fn slice<'a>(&'_ self, slice: &'a GpuSlice<T>) -> Option<&'a GpuSlice<T>> {
		ops::Range{start: 0, end: slice.len() }.slice(slice)
	}
	fn slice_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> Option<&'a mut GpuSlice<T>> {
		ops::Range{start: 0, end: slice.len() }.slice_mut(slice)
	}
}
impl<T> GpuSliceRange<T> for ops::RangeInclusive<usize> {
	fn index<'a>(&'_ self, slice: &'a GpuSlice<T>) -> &'a GpuSlice<T> {
		ops::Range{start: *self.start(), end: *self.end() + 1 }.index(slice)
	}
	fn index_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> &'a mut GpuSlice<T> {
		ops::Range{start: *self.start(), end: *self.end() + 1 }.index_mut(slice)
	}

	fn slice<'a>(&'_ self, slice: &'a GpuSlice<T>) -> Option<&'a GpuSlice<T>> {
		ops::Range{start: *self.start(), end: *self.end() + 1 }.slice(slice)
	}
	fn slice_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> Option<&'a mut GpuSlice<T>> {
		ops::Range{start: *self.start(), end: *self.end() + 1 }.slice_mut(slice)
	}
}
impl<T> GpuSliceRange<T> for ops::RangeToInclusive<usize> {
	fn index<'a>(&'_ self, slice: &'a GpuSlice<T>) -> &'a GpuSlice<T> {
		ops::Range{start: 0, end: self.end + 1 }.index(slice)
	}
	fn index_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> &'a mut GpuSlice<T> {
		ops::Range{start: 0, end: self.end + 1 }.index_mut(slice)
	}

	fn slice<'a>(&'_ self, slice: &'a GpuSlice<T>) -> Option<&'a GpuSlice<T>> {
		ops::Range{start: 0, end: self.end + 1 }.slice(slice)
	}
	fn slice_mut<'a>(&'_ self, slice: &'a mut GpuSlice<T>) -> Option<&'a mut GpuSlice<T>> {
		ops::Range{start: 0, end: self.end + 1 }.slice_mut(slice)
	}
}
