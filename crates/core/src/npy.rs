//! Parsing for the NumPy `.npy` array format (also the payload of each entry in
//! a `.npz` archive). A `.npy` file is a small header — a 6-byte magic, a 2-byte
//! version, a 2- or 4-byte header length, then an ASCII Python-dict describing
//! the array — followed by the raw elements in C (row-major) order, exactly the
//! little-endian layout the rest of the explorer already decodes.
//!
//! See <https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html>.

use std::io::Read;

/// The decoded header of a `.npy` stream.
pub struct NpyHeader {
    /// The explorer dtype name (`F32`, `I16`, …) the descriptor maps to.
    pub dtype: String,
    /// Logical shape (row-major). Reversed for Fortran-order arrays so the raw
    /// bytes still read correctly as a contiguous row-major buffer.
    pub shape: Vec<usize>,
    /// Bytes from the start of the stream to the first data element.
    pub data_offset: usize,
}

/// Read and parse a `.npy` header from the start of `r`. Leaves `r` positioned
/// at the first data byte. Errors on a bad magic or an unsupported dtype.
pub fn parse_header(r: &mut impl Read) -> Result<NpyHeader, String> {
    let mut magic = [0u8; 8]; // 6-byte magic + 2-byte version
    r.read_exact(&mut magic)
        .map_err(|e| format!("reading .npy magic: {e}"))?;
    if &magic[..6] != b"\x93NUMPY" {
        return Err("not a .npy stream (bad magic)".to_string());
    }
    let major = magic[6];
    // v1 uses a 2-byte header length; v2+ widened it to 4 bytes.
    let (header_len, len_field) = if major >= 2 {
        let mut b = [0u8; 4];
        r.read_exact(&mut b)
            .map_err(|e| format!("reading .npy header length: {e}"))?;
        (u32::from_le_bytes(b) as usize, 4)
    } else {
        let mut b = [0u8; 2];
        r.read_exact(&mut b)
            .map_err(|e| format!("reading .npy header length: {e}"))?;
        (u16::from_le_bytes(b) as usize, 2)
    };
    let mut buf = vec![0u8; header_len];
    r.read_exact(&mut buf)
        .map_err(|e| format!("reading .npy header: {e}"))?;
    let header = String::from_utf8_lossy(&buf);

    let descr = dict_string(&header, "descr")?;
    let dtype = map_descr(&descr)?;
    let mut shape = dict_shape(&header)?;
    if dict_bool(&header, "fortran_order")? {
        // Column-major bytes are the row-major bytes of the transposed shape;
        // reversing the dims lets the row-major readers serve correct values.
        shape.reverse();
    }
    Ok(NpyHeader {
        dtype,
        shape,
        data_offset: 8 + len_field + header_len,
    })
}

/// Map a NumPy dtype descriptor (array-interface `typestr`, e.g. `<f4`, `|u1`,
/// `=i8`) to the explorer's dtype name. Rejects big-endian multi-byte types
/// (the decoders assume little-endian) and non-numeric kinds.
pub fn map_descr(descr: &str) -> Result<String, String> {
    let (order, rest) = match descr.as_bytes().first() {
        Some(b'<' | b'=' | b'>' | b'|') => (descr.as_bytes()[0], &descr[1..]),
        _ => (b'=', descr),
    };
    let kind = rest.chars().next().ok_or("empty dtype descriptor")?;
    let size: usize = rest[kind.len_utf8()..]
        .parse()
        .map_err(|_| format!("unsupported dtype: {descr}"))?;
    if order == b'>' && size > 1 {
        return Err(format!("big-endian dtype not supported: {descr}"));
    }
    let name = match (kind, size) {
        ('f', 8) => "F64",
        ('f', 4) => "F32",
        ('f', 2) => "F16",
        ('i', 8) => "I64",
        ('i', 4) => "I32",
        ('i', 2) => "I16",
        ('i', 1) => "I8",
        ('u', 8) => "U64",
        ('u', 4) => "U32",
        ('u', 2) => "U16",
        ('u', 1) => "U8",
        ('b', 1) => "BOOL",
        _ => return Err(format!("unsupported dtype: {descr}")),
    };
    Ok(name.to_string())
}

/// The single-quoted string value of `'key'` in the header dict.
fn dict_string(header: &str, key: &str) -> Result<String, String> {
    let rest = after_key(header, key)?;
    let open = rest.find('\'').ok_or_else(|| missing(key))?;
    let tail = &rest[open + 1..];
    let close = tail.find('\'').ok_or_else(|| missing(key))?;
    Ok(tail[..close].to_string())
}

/// The `True`/`False` value of `'key'` in the header dict.
fn dict_bool(header: &str, key: &str) -> Result<bool, String> {
    let rest = after_key(header, key)?;
    let t = rest.find("True");
    let f = rest.find("False");
    match (t, f) {
        (Some(ti), f) if f.is_none_or(|fi| ti < fi) => Ok(true),
        (_, Some(_)) => Ok(false),
        _ => Err(format!("malformed '{key}' in .npy header")),
    }
}

/// The `'shape'` tuple, e.g. `(4, 5)` → `[4, 5]`, `(5,)` → `[5]`, `()` → `[]`.
fn dict_shape(header: &str) -> Result<Vec<usize>, String> {
    let rest = after_key(header, "shape")?;
    let open = rest.find('(').ok_or_else(|| missing("shape"))?;
    let close = rest[open..].find(')').ok_or_else(|| missing("shape"))? + open;
    rest[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|_| format!("bad dimension in .npy shape: {s}"))
        })
        .collect()
}

/// The slice of the header following `'key':`.
fn after_key(header: &str, key: &str) -> Result<String, String> {
    let pat = format!("'{key}'");
    let at = header.find(&pat).ok_or_else(|| missing(key))?;
    let after = &header[at + pat.len()..];
    let colon = after.find(':').ok_or_else(|| missing(key))?;
    Ok(after[colon + 1..].to_string())
}

fn missing(key: &str) -> String {
    format!("missing '{key}' in .npy header")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_dtype_descriptors() {
        assert_eq!(map_descr("<f4").unwrap(), "F32");
        assert_eq!(map_descr("<f8").unwrap(), "F64");
        assert_eq!(map_descr("<f2").unwrap(), "F16");
        assert_eq!(map_descr("<i2").unwrap(), "I16");
        assert_eq!(map_descr("|i1").unwrap(), "I8");
        assert_eq!(map_descr("|u1").unwrap(), "U8");
        assert_eq!(map_descr("<u4").unwrap(), "U32");
        assert_eq!(map_descr("|b1").unwrap(), "BOOL");
        assert_eq!(map_descr("=i8").unwrap(), "I64");
        // Big-endian multi-byte and exotic kinds are rejected.
        assert!(map_descr(">f4").is_err());
        assert!(map_descr("<c8").is_err());
        assert!(map_descr("<U5").is_err());
    }

    #[test]
    fn parses_a_v1_header() {
        // A real v1.0 header for a 4×5 little-endian f32 C-order array.
        let dict = b"{'descr': '<f4', 'fortran_order': False, 'shape': (4, 5), }";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
        bytes.extend_from_slice(&(dict.len() as u16).to_le_bytes());
        bytes.extend_from_slice(dict);
        let mut cur = std::io::Cursor::new(&bytes);
        let h = parse_header(&mut cur).unwrap();
        assert_eq!(h.dtype, "F32");
        assert_eq!(h.shape, vec![4, 5]);
        assert_eq!(h.data_offset, bytes.len());
    }

    #[test]
    fn fortran_order_reverses_the_shape() {
        let dict = b"{'descr': '<f8', 'fortran_order': True, 'shape': (2, 3), }";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
        bytes.extend_from_slice(&(dict.len() as u16).to_le_bytes());
        bytes.extend_from_slice(dict);
        let h = parse_header(&mut std::io::Cursor::new(&bytes)).unwrap();
        assert_eq!(h.shape, vec![3, 2]);
    }

    #[test]
    fn parses_scalar_and_1d_shapes() {
        assert_eq!(dict_shape("'shape': (), ").unwrap(), Vec::<usize>::new());
        assert_eq!(dict_shape("'shape': (7,), ").unwrap(), vec![7]);
        assert_eq!(dict_shape("'shape': (2, 3, 4), ").unwrap(), vec![2, 3, 4]);
    }
}
