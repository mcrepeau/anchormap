//! `anchormap` — a concurrent hash map with lock-free reads.
//!
//! Uses flat metadata probing (Swiss-table style) with per-segment geometric
//! growth. Elements are **never reordered after insertion** — a value placed
//! at slot `i` stays there for the lifetime of its segment. This enables
//! [`HashMap::get`] to run lock-free from any number of threads concurrently,
//! even while [`HashMap::insert`] is running.
//!
//! # Quick start
//!
//! ```
//! use std::sync::Arc;
//! use anchormap::HashMap;
//!
//! let map = Arc::new(HashMap::<&str, u32>::new(64));
//! map.insert("hello", 42);
//!
//! let map2 = Arc::clone(&map);
//! let handle = std::thread::spawn(move || map2.get(&"hello").copied());
//! assert_eq!(handle.join().unwrap(), Some(42));
//! ```

mod hashing;
mod concurrent;
#[cfg(feature = "serde")]
mod serde;

pub use concurrent::HashMap;
pub use concurrent::{Entry, OccupiedEntry, VacantEntry};
pub use concurrent::Drain;
pub use hashing::{TableHasher, AHashHasher, StandardHasher};
