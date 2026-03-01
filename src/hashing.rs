//! Hashing infrastructure — the [`TableHasher`] trait and its built-in impls.

use std::hash::{BuildHasher, Hash};

/// Trait for table hashing — allows different hash algorithms to be used.
///
/// A blanket implementation is provided for all `S: BuildHasher + Clone`, so
/// any standard-library or third-party build-hasher (e.g.
/// `std::collections::hash_map::RandomState`, `ahash::RandomState`) can be
/// passed directly to [`HashMap::with_capacity_and_hasher`](crate::HashMap::with_capacity_and_hasher).
pub trait TableHasher: Clone {
    /// Compute a 64-bit hash for the given key.
    fn hash_key<K: Hash + ?Sized>(&self, key: &K) -> u64;
}

/// Blanket impl: any `BuildHasher + Clone` is automatically a `TableHasher`.
impl<S: BuildHasher + Clone> TableHasher for S {
    #[inline]
    fn hash_key<K: Hash + ?Sized>(&self, key: &K) -> u64 {
        self.hash_one(key)
    }
}

/// AHash-based hasher — the default for [`HashMap`](crate::HashMap).
///
/// AHash is a high-performance, non-cryptographic hash function that is also
/// the default hasher used by [Hashbrown](https://github.com/rust-lang/hashbrown),
/// making benchmark comparisons between the two tables fair and direct.
///
/// Each instance stores its own [`ahash::RandomState`] (seeded from OS entropy
/// at construction), so `hash_key` is a plain `hash_one` call with no global
/// reads.  Hashes are consistent within a single map but differ between maps
/// created in separate calls to `new()`.
#[derive(Clone)]
pub struct AHashHasher(ahash::RandomState);

impl Default for AHashHasher {
    fn default() -> Self {
        Self(ahash::RandomState::new())
    }
}

impl std::fmt::Debug for AHashHasher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AHashHasher").finish_non_exhaustive()
    }
}

impl TableHasher for AHashHasher {
    #[inline]
    fn hash_key<K: Hash + ?Sized>(&self, key: &K) -> u64 {
        self.0.hash_one(key)
    }
}

/// SipHash-based hasher backed by `std::collections::hash_map::RandomState`.
///
/// Randomizes the seed from OS entropy on construction, providing genuine
/// resistance to hash-flooding attacks. Prefer this when keys are
/// externally-controlled and DoS resistance is required.
///
/// ~3–5× slower than `AHashHasher` on typical workloads.
#[derive(Clone)]
pub struct StandardHasher(std::collections::hash_map::RandomState);

impl StandardHasher {
    /// Creates a new `StandardHasher` with a fresh random seed.
    pub fn new() -> Self {
        Self(std::collections::hash_map::RandomState::new())
    }
}

impl Default for StandardHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for StandardHasher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StandardHasher").finish_non_exhaustive()
    }
}

impl TableHasher for StandardHasher {
    #[inline]
    fn hash_key<K: Hash + ?Sized>(&self, key: &K) -> u64 {
        self.0.hash_one(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ahash_hasher() {
        let hasher = AHashHasher::default();
        let h1 = hasher.hash_key(&"test");
        let h2 = hasher.hash_key(&"test");
        assert_eq!(h1, h2); // Same input → same hash within a process run

        let h3 = hasher.hash_key(&"different");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_standard_hasher() {
        let hasher = StandardHasher::new();
        let h1 = hasher.hash_key(&"test");
        let h2 = hasher.hash_key(&"test");
        assert_eq!(h1, h2);

        let h3 = hasher.hash_key(&"different");
        assert_ne!(h1, h3);
    }
}
