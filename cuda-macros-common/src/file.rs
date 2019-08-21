//! File input/output helper module

use std::io::{self, Read, Write, Seek, SeekFrom};

/// Trait that represents a file-like object.
pub trait FileLike: Write + Read + Seek {}
impl<T> FileLike for T where T: Write + Read + Seek {}

/// Helper struct that allows for automatic indentation to a file.
/// 
/// Indentation is tabs, not spaces.
/// 
/// # Examples
/// ```
/// # use cuda_macros_common::file::FileLikeIndent;
/// use std::io::{self, prelude::*};
/// let mut out = vec![];
/// {
/// 	let mut out = io::Cursor::new(&mut out);
/// 	let mut out = FileLikeIndent::new(&mut out, 0);
/// 	writeln!(&mut out, "foo: {{");
/// 	{
/// 		let mut out = out.incr();
/// 		writeln!(&mut out, "testing");
/// 		writeln!(&mut out, "123");
/// 	}
/// 	writeln!(&mut out, "}}");
/// }
/// assert_eq!{out.as_slice(), "\
/// foo: {
/// 	testing
/// 	123
/// }
/// ".as_bytes()}
/// ```
pub struct FileLikeIndent<'f, F: FileLike> {
	f: &'f mut F,
	indent: usize,
}
impl<'f, F> FileLikeIndent<'f, F> where F: FileLike {
	/// Constructs a new `FileLikeIndent` with the specified output file and indentation.
	pub fn new(f: &'f mut F, indent: usize) -> Self {
		FileLikeIndent{ f, indent }
	}
	/// Returns a new `FileLikeIndent` object with the indent incremented by 1.
	pub fn incr<'a, 'b>(&'a mut self) -> FileLikeIndent<'b, F> where 'a: 'b, 'f: 'a {
		FileLikeIndent::new(self.f, self.indent + 1)
	}
	/// Returns a new `FileLikeIndent` object with the indent decremented by 1.
	pub fn decr<'a, 'b>(&'a mut self) -> FileLikeIndent<'b, F> where 'a: 'b, 'f: 'a {
		FileLikeIndent::new(self.f, self.indent.saturating_sub(1))
	}
	/// Returns a new `FileLikeIndent` object with the indent kept the same.
	pub fn clone<'a, 'b>(&'a mut self) -> FileLikeIndent<'b, F> where 'a: 'b, 'f: 'a {
		FileLikeIndent::new(self.f, self.indent)
	}
	/// Get an immutable reference to the underlying `T: FileLike` object.
	pub fn get_ref(&self) -> &F {
		&self.f
	}
	/// Get a mutable reference to the underlying `T: FileLike` object.
	pub fn get_mut(&mut self) -> &mut F {
		&mut self.f
	}

	fn write_indent(&mut self) -> io::Result<()> {
		for _ in 0..self.indent {
			self.f.write(b"\t")?;
		}
		Ok(())
	}

	fn write_buf_no_newline(&mut self, buf: &[u8]) -> io::Result<usize> {
		let mut temp = [0];
		let pos = self.seek(SeekFrom::Current(0))?;
		let num = if pos == 0 {
			0
		} else {
			self.seek(SeekFrom::Current(-1))?;
			self.read(&mut temp)?
		};
		if num == 1 && temp[0] == b'\n' {
			self.write_indent()?;
		}
		self.f.write(buf)
	}
}
impl<'f, F> Write for FileLikeIndent<'f, F> where F: FileLike {
	fn write(&mut self, mut buf: &[u8]) -> io::Result<usize> {
		let mut total = 0;
		while buf.len() > 0 {
			let num = if let Some(i) = buf.iter().position(|&b| b == b'\n') {
				self.write_buf_no_newline(&buf[..=i])?
			} else {
				self.write_buf_no_newline(buf)?
			};
			if num == 0 {
				break;
			}
			buf = &buf[num..];
			total += num;
		}
		Ok(total)
	}
	fn flush(&mut self) -> io::Result<()> {
		self.f.flush()
	}
}
impl<'f, F> Read for FileLikeIndent<'f, F> where F: FileLike {
	fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
		self.f.read(buf)
	}
}
impl<'f, F> Seek for FileLikeIndent<'f, F> where F: FileLike {
	// TODO: fix seeking so that it ignores inserted indentation
	fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
		self.f.seek(pos)
	}
}
impl<'f, F: FileLike> From<&'f mut F> for FileLikeIndent<'f, F> {
	fn from(f: &'f mut F) -> Self {
		FileLikeIndent::new(f, 0)
	}
}

#[cfg(test)]
mod tests {
	use std::io::Cursor;

	use super::*;

	#[test]
	fn test_file_like_indent() {
		let mut f = Cursor::new(Vec::new());
		{
			let mut fi = FileLikeIndent::new(&mut f, 0);
			fi.write(b"one").unwrap();
			assert_eq!(&fi.get_ref().get_ref()[..], b"one");
			fi.write(b"\n").unwrap();
			assert_eq!(&fi.get_ref().get_ref()[..], b"one\n");
			fi.write(b"two").unwrap();
			assert_eq!(&fi.get_ref().get_ref()[..], b"one\ntwo");
			{
				let mut fi = fi.incr();
				fi.write(b"\n").unwrap();
				assert_eq!(&fi.get_ref().get_ref()[..], b"one\ntwo\n");
				fi.write(b"t").unwrap();
				assert_eq!(&fi.get_ref().get_ref()[..], b"one\ntwo\n\tt");
				fi.write(b"hree\n").unwrap();
				assert_eq!(&fi.get_ref().get_ref()[..], b"one\ntwo\n\tthree\n");
				{
					let mut fi = fi.incr();
					assert_eq!(&fi.get_ref().get_ref()[..], b"one\ntwo\n\tthree\n");
					fi.write(b"four").unwrap();
					assert_eq!(&fi.get_ref().get_ref()[..], b"one\ntwo\n\tthree\n\t\tfour");
					{
						let mut fi = fi.incr();
						fi.write(b"\n").unwrap();
					}
					assert_eq!(&fi.get_ref().get_ref()[..], b"one\ntwo\n\tthree\n\t\tfour\n");
					fi.write(b"five").unwrap();
					assert_eq!(&fi.get_ref().get_ref()[..], b"one\ntwo\n\tthree\n\t\tfour\n\t\tfive");
					fi.write(b"\n").unwrap();
					assert_eq!(&fi.get_ref().get_ref()[..], b"one\ntwo\n\tthree\n\t\tfour\n\t\tfive\n");
				}
				fi.write(b"six").unwrap();
				assert_eq!(&fi.get_ref().get_ref()[..], &b"one\ntwo\n\tthree\n\t\tfour\n\t\tfive\n\tsix"[..]);
			}
			fi.write(b"\nseven").unwrap();
			assert_eq!(&fi.get_ref().get_ref()[..], &b"one\ntwo\n\tthree\n\t\tfour\n\t\tfive\n\tsix\nseven"[..]);
		}
	}
}
