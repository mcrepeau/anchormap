//! Exhaustive Loom models for the concurrent metadata state machine.
//!
//! These scenarios exercise the interleavings that the lock-free reader
//! guarantee rests on: the claim (`META_EMPTY → META_RESERVED`) / publish
//! (Release fingerprint store) sequence, cross-stripe CAS contention and
//! the loser's fall-through, concurrent segment growth and its publication
//! chain (segment pointer Release store → `n_segments` Release fetch_add →
//! reader Acquire), and `remove`'s tombstone transition racing with readers
//! holding `&V`.
//!
//! Run with:
//!
//! ```text
//! RUSTFLAGS="--cfg loom" cargo test --lib loom_tests
//! ```
//!
//! Under `cfg(loom)` the crate's atomics and mutexes are Loom's simulated
//! implementations, so every feasible interleaving of the operations below
//! is enumerated and checked against the C++/Rust memory model. `GROUP_SIZE`
//! and `NUM_STRIPES` are reduced under `cfg(loom)` to keep the explored state
//! space tractable; no algorithmic code changes. Loom validates the atomic
//! state machine and its happens-before reasoning; it does not track raw
//! `(K, V)` memory accesses (that is Miri's domain — both suites are kept).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use loom::thread;

use crate::concurrent::NUM_STRIPES;
use crate::hashing::TableHasher;
use crate::HashMap;

/// Deterministic hasher (fixed-key SipHash) so stripe assignment and slot
/// placement are reproducible across Loom executions.
#[derive(Clone)]
struct FixedHasher;

impl TableHasher for FixedHasher {
    fn hash_key<K: Hash + ?Sized>(&self, key: &K) -> u64 {
        let mut h = DefaultHasher::new();
        key.hash(&mut h);
        h.finish()
    }
}

/// Returns the first key `>= start` whose stripe (`hash % NUM_STRIPES`) is
/// `want`, so scenarios can deliberately place writers in the same or
/// different stripes.
fn key_with_stripe(start: u64, want: usize) -> u64 {
    let hasher = FixedHasher;
    (start..)
        .find(|&k| (hasher.hash_key(&k) as usize) % NUM_STRIPES == want)
        .expect("a key of each stripe exists")
}

/// Minimum-size map: one probe group of `GROUP_SIZE` slots, so every key's
/// probe window covers the same slots.
fn new_map<V>() -> HashMap<u64, V, FixedHasher> {
    HashMap::with_hasher(1, FixedHasher)
}

/// A writer inserts a key while a reader probes for it. The reader must
/// observe either nothing or the fully published value — never the
/// in-flight `META_RESERVED` state or partially visible data.
#[test]
fn claim_publish_reader_vs_writer() {
    loom::model(|| {
        let map = Arc::new(new_map::<u64>());
        let k = key_with_stripe(0, 0);

        let writer = {
            let map = Arc::clone(&map);
            thread::spawn(move || map.insert(k, 42))
        };
        let reader = {
            let map = Arc::clone(&map);
            thread::spawn(move || map.get(&k).copied())
        };

        writer.join().unwrap();
        let seen = reader.join().unwrap();
        assert!(seen.is_none() || seen == Some(42), "reader saw {seen:?}");

        assert_eq!(map.get(&k), Some(&42));
        assert_eq!(map.len(), 1);
    });
}

/// Two writers on different stripes whose probe windows cover the same
/// slots. Both may find the same empty slot; exactly one CAS succeeds and
/// the loser must fall through to `insert_absent` without losing its key.
#[test]
fn cross_stripe_cas_loser() {
    loom::model(|| {
        let map = Arc::new(new_map::<u64>());
        let k0 = key_with_stripe(0, 0);
        let k1 = key_with_stripe(0, 1);
        assert_ne!(k0, k1);

        let a = {
            let map = Arc::clone(&map);
            thread::spawn(move || map.insert(k0, 10))
        };
        let b = {
            let map = Arc::clone(&map);
            thread::spawn(move || map.insert(k1, 20))
        };
        a.join().unwrap();
        b.join().unwrap();

        assert_eq!(map.get(&k0), Some(&10));
        assert_eq!(map.get(&k1), Some(&20));
        assert_eq!(map.len(), 2);
    });
}

/// Segment 0 is full, so two different-stripe writers both exhaust their
/// probe sequences and contend on the growth lock. Exactly one allocates
/// the new segment; both keys must end up present and all anchored values
/// must remain intact. Also exercises the stripe-lock → growth-lock
/// ordering for deadlock-freedom.
#[test]
fn concurrent_growth_two_writers() {
    loom::model(|| {
        let map = Arc::new(new_map::<u64>());
        for i in 0..4u64 {
            assert!(map.insert(i, i * 100));
        }

        let k5 = key_with_stripe(100, 0);
        let k6 = key_with_stripe(200, 1);
        assert_ne!(k5, k6);

        let a = {
            let map = Arc::clone(&map);
            thread::spawn(move || map.insert(k5, 55))
        };
        let b = {
            let map = Arc::clone(&map);
            thread::spawn(move || map.insert(k6, 66))
        };
        a.join().unwrap();
        b.join().unwrap();

        assert_eq!(map.get(&k5), Some(&55));
        assert_eq!(map.get(&k6), Some(&66));
        for i in 0..4u64 {
            assert_eq!(map.get(&i), Some(&(i * 100)));
        }
        assert_eq!(map.len(), 6);
    });
}

/// A reader races a writer that triggers segment growth. Checks the
/// publication chain: a reader that Acquire-loads a `n_segments` value that
/// includes the new segment must be able to see the new segment's pointer,
/// and anchored values in segment 0 stay visible throughout.
#[test]
fn reader_during_growth() {
    loom::model(|| {
        let map = Arc::new(new_map::<u64>());
        for i in 0..4u64 {
            map.insert(i, i * 100);
        }
        let k5 = key_with_stripe(100, 1);

        let writer = {
            let map = Arc::clone(&map);
            thread::spawn(move || map.insert(k5, 55))
        };
        let reader = {
            let map = Arc::clone(&map);
            thread::spawn(move || {
                // Anchored value: always fully visible, never moved.
                assert_eq!(map.get(&0), Some(&0));
                // Growing value: visible only once fully published.
                map.get(&k5).copied()
            })
        };

        writer.join().unwrap();
        let seen = reader.join().unwrap();
        assert!(seen.is_none() || seen == Some(55), "reader saw {seen:?}");

        assert_eq!(map.get(&k5), Some(&55));
        assert_eq!(map.len(), 5);
    });
}

/// A `&V` obtained from `get()` is held across a concurrent remove of the
/// same key plus a different-stripe insert whose probe window covers the
/// tombstoned slot. Tombstones must never be reclaimed under `&self`: the
/// held reference must still read the original value afterwards.
#[test]
fn tombstone_ref_stability() {
    loom::model(|| {
        let map = Arc::new(new_map::<Box<u64>>());
        let k = key_with_stripe(0, 0);
        map.insert(k, Box::new(7));
        let r = map.get(&k).expect("inserted above");

        let k2 = key_with_stripe(100, 1);
        let writer = {
            let map = Arc::clone(&map);
            thread::spawn(move || {
                assert!(map.remove(&k));
                assert!(map.insert(k2, Box::new(8)));
            })
        };
        writer.join().unwrap();

        // The held reference must be unaffected: the removed value is never
        // dropped or overwritten while the map is alive through &self.
        assert_eq!(**r, 7);
        assert_eq!(map.get(&k), None);
        assert_eq!(map.get(&k2).map(|b| **b), Some(8));
        assert_eq!(map.len(), 1);
    });
}

/// Two threads remove the same key concurrently (same stripe, so they
/// serialize on the stripe lock). Exactly one removal must succeed —
/// exercises the O(1) hint verify and its full-rescan fallback.
#[test]
fn double_remove_same_key() {
    loom::model(|| {
        let map = Arc::new(new_map::<u64>());
        let k = key_with_stripe(0, 0);
        map.insert(k, 7);

        let a = {
            let map = Arc::clone(&map);
            thread::spawn(move || map.remove(&k))
        };
        let b = {
            let map = Arc::clone(&map);
            thread::spawn(move || map.remove(&k))
        };
        let ra = a.join().unwrap();
        let rb = b.join().unwrap();

        assert_ne!(ra, rb, "exactly one remove must succeed");
        assert_eq!(map.get(&k), None);
        assert_eq!(map.len(), 0);
    });
}

/// Two threads `get_or_insert` the same key: both must observe the same
/// value and exactly one insertion must occur.
#[test]
fn get_or_insert_dedup() {
    loom::model(|| {
        let map = Arc::new(new_map::<u64>());
        let k = key_with_stripe(0, 0);

        let a = {
            let map = Arc::clone(&map);
            thread::spawn(move || *map.get_or_insert(k, 7))
        };
        let b = {
            let map = Arc::clone(&map);
            thread::spawn(move || *map.get_or_insert(k, 7))
        };
        let va = a.join().unwrap();
        let vb = b.join().unwrap();

        assert_eq!(va, 7);
        assert_eq!(vb, 7);
        assert_eq!(map.len(), 1);
    });
}
