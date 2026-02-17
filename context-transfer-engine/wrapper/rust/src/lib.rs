mod ffi_c;

#[cxx::bridge(namespace = "cte_ffi")]
mod ffi {
    struct CteTagId {
        major: u32,
        minor: u32,
    }

    unsafe extern "C++" {
        include!("shim/shim.h");

        type CteTag;

        fn cte_init(config_path: &str) -> bool;
        fn tag_new(tag_name: &str) -> UniquePtr<CteTag>;
        fn tag_from_id(major: u32, minor: u32) -> UniquePtr<CteTag>;
        fn tag_put_blob(tag: &CteTag, name: &str, data: &[u8], offset: u64, score: f32);
        fn tag_get_blob(
            tag: &CteTag,
            name: &str,
            size: u64,
            offset: u64,
        ) -> UniquePtr<CxxVector<u8>>;
        fn tag_get_blob_score(tag: &CteTag, name: &str) -> f32;
        fn tag_get_blob_size(tag: &CteTag, name: &str) -> u64;
        fn tag_get_contained_blobs(tag: &CteTag) -> UniquePtr<CxxVector<CxxString>>;
        fn tag_reorganize_blob(tag: &CteTag, name: &str, score: f32);
        fn tag_get_id(tag: &CteTag) -> CteTagId;
        fn client_register_target(target_path: &str, size: u64) -> bool;
        fn client_del_tag(name: &str) -> bool;
        fn client_tag_query(regex: &str, max_tags: u32) -> UniquePtr<CxxVector<CxxString>>;
        fn client_blob_query(
            tag_re: &str,
            blob_re: &str,
            max_results: u32,
        ) -> UniquePtr<CxxVector<CxxString>>;
    }
}

pub use ffi::CteTagId;

/// Initialize CTE with an embedded runtime.
///
/// Must be called once before any other CTE operations.
/// `config_path` can be empty to use default configuration.
pub fn init(config_path: &str) -> Result<(), String> {
    if ffi::cte_init(config_path) {
        Ok(())
    } else {
        Err("CTE initialization failed".into())
    }
}

/// A handle to a CTE tag (bucket / container).
pub struct Tag {
    inner: cxx::UniquePtr<ffi::CteTag>,
}

impl Tag {
    /// Create or get a tag by name.
    pub fn new(name: &str) -> Self {
        Self {
            inner: ffi::tag_new(name),
        }
    }

    /// Open an existing tag by its ID.
    pub fn from_id(id: CteTagId) -> Self {
        Self {
            inner: ffi::tag_from_id(id.major, id.minor),
        }
    }

    /// Write data into a blob with default offset (0) and score (1.0).
    pub fn put_blob(&self, name: &str, data: &[u8]) {
        ffi::tag_put_blob(&self.inner, name, data, 0, 1.0);
    }

    /// Write data into a blob with explicit offset and score.
    pub fn put_blob_with_options(&self, name: &str, data: &[u8], offset: u64, score: f32) {
        ffi::tag_put_blob(&self.inner, name, data, offset, score);
    }

    /// Read blob data. Returns a `Vec<u8>` of `size` bytes starting at `offset`.
    pub fn get_blob(&self, name: &str, size: u64) -> Vec<u8> {
        let v = ffi::tag_get_blob(&self.inner, name, size, 0);
        v.iter().copied().collect()
    }

    /// Read blob data with explicit offset.
    pub fn get_blob_with_offset(&self, name: &str, size: u64, offset: u64) -> Vec<u8> {
        let v = ffi::tag_get_blob(&self.inner, name, size, offset);
        v.iter().copied().collect()
    }

    /// Get the placement score of a blob.
    pub fn get_blob_score(&self, name: &str) -> f32 {
        ffi::tag_get_blob_score(&self.inner, name)
    }

    /// Get the size of a blob in bytes.
    pub fn get_blob_size(&self, name: &str) -> u64 {
        ffi::tag_get_blob_size(&self.inner, name)
    }

    /// List all blob names in this tag.
    pub fn get_contained_blobs(&self) -> Vec<String> {
        let v = ffi::tag_get_contained_blobs(&self.inner);
        v.iter().map(|s| s.to_string_lossy().into_owned()).collect()
    }

    /// Change the placement score of a blob, triggering data migration.
    pub fn reorganize_blob(&self, name: &str, score: f32) {
        ffi::tag_reorganize_blob(&self.inner, name, score);
    }

    /// Get the tag's unique ID.
    pub fn get_tag_id(&self) -> CteTagId {
        ffi::tag_get_id(&self.inner)
    }
}

/// Static client operations (no tag context needed).
pub struct Client;

impl Client {
    /// Register a file-backed storage target with the CTE pool.
    pub fn register_target(target_path: &str, size: u64) -> bool {
        ffi::client_register_target(target_path, size)
    }

    /// Delete a tag by name.
    pub fn del_tag(name: &str) -> bool {
        ffi::client_del_tag(name)
    }

    /// Query tags matching a regex pattern.
    pub fn tag_query(regex: &str, max_tags: u32) -> Vec<String> {
        let v = ffi::client_tag_query(regex, max_tags);
        v.iter().map(|s| s.to_string_lossy().into_owned()).collect()
    }

    /// Query blobs matching tag and blob regex patterns.
    /// Returns pairs of (tag_name, blob_name).
    pub fn blob_query(tag_re: &str, blob_re: &str, max_results: u32) -> Vec<(String, String)> {
        let v = ffi::client_blob_query(tag_re, blob_re, max_results);
        let flat: Vec<String> = v.iter().map(|s| s.to_string_lossy().into_owned()).collect();
        flat.chunks(2)
            .filter_map(|c| {
                if c.len() == 2 {
                    Some((c[0].clone(), c[1].clone()))
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_and_roundtrip() {
        init("").expect("CTE init failed");

        // Register a file-backed storage target (required for PutBlob)
        let target_path = "/tmp/cte_rust_test_target";
        Client::register_target(target_path, 64 * 1024 * 1024);
        // Allow target registration to propagate
        std::thread::sleep(std::time::Duration::from_millis(200));

        let tag = Tag::new("rust_test_tag");
        let id = tag.get_tag_id();
        assert!(id.major != 0 || id.minor != 0, "tag ID should be non-null");

        let data = b"hello from rust";
        tag.put_blob("test_blob", data);

        let size = tag.get_blob_size("test_blob");
        assert_eq!(size, data.len() as u64);

        let got = tag.get_blob("test_blob", size);
        assert_eq!(got, data);

        let blobs = tag.get_contained_blobs();
        assert!(blobs.contains(&"test_blob".to_string()));

        Client::del_tag("rust_test_tag");
    }

    #[test]
    fn test_config_based_init() {
        // Use CHI_SERVER_CONF like the memorybench does
        std::env::set_var(
            "CHI_SERVER_CONF",
            "/workspace/context-transfer-engine/benchmark/memorybench/cte_config.yaml",
        );
        std::env::set_var("CHI_WITH_RUNTIME", "1");

        init("").expect("CTE init failed");

        let tag = Tag::new("config_test_tag");
        let id = tag.get_tag_id();
        eprintln!("tag id: major={}, minor={}", id.major, id.minor);
        assert!(id.major != 0 || id.minor != 0, "tag ID should be non-null");

        let data = b"hello from config test";
        tag.put_blob("test_blob", data);

        let size = tag.get_blob_size("test_blob");
        assert_eq!(size, data.len() as u64);

        let got = tag.get_blob("test_blob", size);
        assert_eq!(got, data);

        Client::del_tag("config_test_tag");
    }
}
