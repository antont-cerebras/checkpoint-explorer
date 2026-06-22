//! Throwaway diagnostic: dump shape / dtype / chunk shape / filters / on-disk
//! size for datasets in an HDF5 file (optionally filtered by a name substring).
//!   cargo run --example h5layout --features hdf5 -- <file.hdf5> [name-substr]
use hdf5_metno::{File, Group};

fn walk(g: &Group, want: Option<&str>) {
    for ds in g.datasets().unwrap_or_default() {
        let name = ds.name();
        if want.is_some_and(|w| !name.contains(w)) {
            continue;
        }
        let shape = ds.shape();
        let logical: usize =
            shape.iter().product::<usize>() * ds.dtype().map(|d| d.size()).unwrap_or(0);
        let stored = ds.storage_size() as usize;
        let chunk = ds.chunk();
        let filters: Vec<String> = ds.filters().iter().map(|f| format!("{f:?}")).collect();
        println!("{name}");
        println!("  shape        {shape:?}");
        println!(
            "  dtype size   {} B",
            ds.dtype().map(|d| d.size()).unwrap_or(0)
        );
        println!(
            "  chunk        {chunk:?}  (num_chunks {:?})",
            ds.num_chunks()
        );
        println!("  filters      {filters:?}");
        println!(
            "  logical {:.2} GiB  on-disk {:.2} GiB  ratio {:.2}x",
            logical as f64 / (1u64 << 30) as f64,
            stored as f64 / (1u64 << 30) as f64,
            if stored > 0 {
                logical as f64 / stored as f64
            } else {
                0.0
            }
        );
    }
    for sub in g.groups().unwrap_or_default() {
        walk(&sub, want);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: h5layout <file> [name-substr]");
    let want = args.get(2).map(|s| s.as_str());
    let f = File::open(path).expect("open");
    walk(&f, want);
}
