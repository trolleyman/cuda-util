
use std::io::{self, Read, Write, Seek, SeekFrom};

/// Trait that represents a file-like object
pub trait FileLike: Write + Read + Seek {}
impl<T> FileLike for T where T: Write + Read + Seek {}

/// Helper struct that allows for automatic indentation to a file
pub struct FileLikeIndent<'f> {
	f: &'f mut FileLike,
	indent: usize,
}
impl<'f> FileLikeIndent<'f> {
	pub fn new(f: &'f mut FileLike, indent: usize) -> Self {
		FileLikeIndent{ f, indent }
	}
	pub fn incr<'a, 'b>(&'a mut self) -> FileLikeIndent<'b> where 'a: 'b, 'f: 'a {
		FileLikeIndent::new(self.f, self.indent + 1)
	}
	pub fn decr<'a, 'b>(&'a mut self) -> FileLikeIndent<'b> where 'a: 'b, 'f: 'a {
		FileLikeIndent::new(self.f, self.indent.saturating_sub(1))
	}

	fn write_indent(&mut self) -> io::Result<()> {
		for _ in 0..self.indent {
			self.f.write(b"\t")?;
		}
		Ok(())
	}

	fn write_buf_no_indent(&mut self, buf: &[u8]) -> io::Result<usize> {
		self.f.write(buf)
	}
}
impl<'f> Write for FileLikeIndent<'f> {
	fn write(&mut self, mut buf: &[u8]) -> io::Result<usize> {
		let mut total = 0;
		while buf.len() > 0 {
			let num = if let Some(i) = buf.iter().cloned().position(|b| b == b'\n') {
				let num = self.write_buf_no_indent(&buf[..=i])?;
				if num != 0 {
					self.write_indent()?;
				}
				num
			} else {
				self.write_buf_no_indent(buf)?
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
impl<'f> Read for FileLikeIndent<'f> {
	fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
		self.f.read(buf)
	}
}
impl<'f> Seek for FileLikeIndent<'f> {
	// TODO: fix seeking so that it ignores inserted indentation
	fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
		self.f.seek(pos)
	}
}
