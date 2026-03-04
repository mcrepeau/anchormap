//! `anchormap` — a concurrent hash map with lock-free reads.
//!
//! Uses flat metadata probing (Swiss-table style) with per-segment geometric
//! growth. Under shared (`&self`) access, an inserted value **never moves**:
//! a value placed at slot `i` stays there for the lifetime of its segment,
//! which is why [`HashMap::get`] can run lock-free from any number of threads
//! even while [`HashMap::insert`] is running.
//!
//! Exclusive (`&mut self`) operations such as [`HashMap::shrink_to_fit`] may
//! relocate entries, but the borrow checker guarantees that no `&V` reference
//! obtained from a prior `get()` call can be held across such an operation.
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
pub use concurrent::{Drain, IterMut};
pub use hashing::{TableHasher, AHashHasher, StandardHasher};
