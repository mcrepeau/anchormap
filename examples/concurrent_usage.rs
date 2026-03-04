//! Demonstration of `anchormap::HashMap`'s API.
//!
//! Works through the key features in roughly the order a new user would
//! encounter them. Each section is self-contained and prints its results.
//!
//! Run with:
//!   cargo run --example concurrent_usage

use std::sync::Arc;
use anchormap::{HashMap, StandardHasher};

fn main() {
    basic_operations();
    lock_free_reference_lifetime();
    concurrent_reads();
    concurrent_writes_and_reads();
    find_or_create();
    entry_api();
    exclusive_mutation();
    capacity_management();
    iteration_and_drain();
    collection_traits();
    custom_hasher();
}

// ── 1. Basic operations ───────────────────────────────────────────────────────

fn basic_operations() {
    println!("── 1. Basic operations ──");

    let map = HashMap::<&str, u32>::new(64);

    // insert returns true on a new key, false if already present.
    assert!(map.insert("alpha", 1));
    assert!(map.insert("beta",  2));
    assert!(!map.insert("alpha", 99)); // duplicate — map is unchanged

    assert_eq!(map.get(&"alpha"), Some(&1));
    assert_eq!(map.get(&"gamma"), None);
    assert!(map.contains_key(&"beta"));

    println!("  len={}, capacity={}", map.len(), map.capacity());

    // remove() tombstones the slot; outstanding &V references remain valid.
    assert!(map.remove(&"beta"));
    assert!(!map.remove(&"beta")); // already gone

    println!("  after remove: len={}", map.len());
}

// ── 2. Lock-free reference lifetime ──────────────────────────────────────────

fn lock_free_reference_lifetime() {
    println!("\n── 2. Lock-free reference lifetime ──");

    // The defining feature of anchormap: get() returns a plain &V with no
    // guard.  The reference is valid for the entire shared-borrow lifetime of
    // the map — even while other threads (or this thread) call insert().
    //
    // With DashMap, holding a Ref across an insert on the same shard would
    // deadlock.  With Papaya, you must keep an epoch guard alive.  Here,
    // nothing extra is needed.

    let map = HashMap::<&str, Vec<u32>>::new(64);
    map.insert("primes", vec![2, 3, 5, 7, 11]);

    // Borrow the value.
    let primes: &Vec<u32> = map.get(&"primes").unwrap();

    // Insert more keys while holding the reference — no conflict.
    map.insert("evens", vec![2, 4, 6, 8]);
    map.insert("odds",  vec![1, 3, 5, 7]);

    // primes is still valid.
    println!("  primes (held across two inserts): {:?}", primes);

    // The same works across thread boundaries via Arc.
    let map = Arc::new(HashMap::<u32, String>::new(64));
    map.insert(1, "one".to_string());

    let v: &String = map.get(&1).unwrap();
    let map2 = Arc::clone(&map);

    // Spawn a writer — the reference v remains valid on this thread.
    let handle = std::thread::spawn(move || {
        map2.insert(2, "two".to_string());
    });
    handle.join().unwrap();

    println!("  value held across concurrent insert: \"{}\"", v);
}

// ── 3. Concurrent reads ───────────────────────────────────────────────────────

fn concurrent_reads() {
    println!("\n── 3. Concurrent reads (8 threads) ──");

    let map = Arc::new(HashMap::<u32, u32>::new(512));
    for i in 0..200u32 { map.insert(i, i * 10); }

    std::thread::scope(|s| {
        for t in 0..8 {
            let map = Arc::clone(&map);
            s.spawn(move || {
                let found = (0..200u32)
                    .filter(|i| map.get(i) == Some(&(*i * 10)))
                    .count();
                println!("  thread {t}: verified {found}/200 entries");
            });
        }
    });
}

// ── 4. Concurrent writes + reads ─────────────────────────────────────────────

fn concurrent_writes_and_reads() {
    println!("\n── 4. Concurrent writes + reads ──");

    let map = Arc::new(HashMap::<u32, String>::new(512));

    std::thread::scope(|s| {
        // 4 writer threads, each owning a disjoint key range.
        for t in 0..4u32 {
            let map = Arc::clone(&map);
            s.spawn(move || {
                for i in (t * 50)..(t * 50 + 50) {
                    map.insert(i, format!("v{i}"));
                }
            });
        }
        // 2 reader threads scanning the entire range simultaneously.
        for r in 0..2u32 {
            let map = Arc::clone(&map);
            s.spawn(move || {
                let seen = (0..200u32).filter(|i| map.get(i).is_some()).count();
                println!("  reader {r}: observed {seen}/200 (snapshot mid-write)");
            });
        }
    });

    println!("  final size: {}", map.len());
}

// ── 5. Find-or-create ─────────────────────────────────────────────────────────

fn find_or_create() {
    println!("\n── 5. Find-or-create ──");

    // get_or_insert / get_or_insert_with are available on &self, so they can
    // be called concurrently without &mut.  The first caller wins; all others
    // get back the same &V.
    let map = Arc::new(HashMap::<&str, u32>::new(64));

    std::thread::scope(|s| {
        for _ in 0..8 {
            let map = Arc::clone(&map);
            s.spawn(move || {
                let v = map.get_or_insert("counter", 0);
                assert_eq!(*v, 0); // exactly one thread inserted; all see 0
            });
        }
    });
    println!("  get_or_insert with 8 racing threads: value = {}", map[&"counter"]);

    // get_or_insert_with — closure called only if key absent.
    let v = map.get_or_insert_with("expensive", || {
        // In practice this might be a DB lookup, a parse, etc.
        42u32
    });
    println!("  get_or_insert_with (absent): {v}");

    let v2 = map.get_or_insert_with("expensive", || panic!("should not be called"));
    println!("  get_or_insert_with (present, closure skipped): {v2}");

    // get_or_try_insert_with — propagates Err without modifying the map.
    let ok:  Result<&u32, &str> = map.get_or_try_insert_with("new_key", || Ok(7));
    let err: Result<&u32, &str> = map.get_or_try_insert_with("fail_key", || Err("bad input"));
    println!("  get_or_try_insert_with ok={ok:?}  err={err:?}");
    assert!(!map.contains_key(&"fail_key")); // map unchanged on Err
}

// ── 6. Entry API ──────────────────────────────────────────────────────────────

fn entry_api() {
    println!("\n── 6. Entry API ──");

    let mut map = HashMap::<&str, u32>::new(64);

    // or_insert — insert default if absent, return &mut V either way.
    *map.entry("hits").or_insert(0) += 1;
    *map.entry("hits").or_insert(0) += 1;
    println!("  hits after two increments: {}", map[&"hits"]);

    // and_modify + or_insert — update if present, insert if absent.
    map.entry("score")
        .and_modify(|v| *v += 10)
        .or_insert(10);
    map.entry("score")
        .and_modify(|v| *v += 10)
        .or_insert(10);
    println!("  score: {}", map[&"score"]); // 20

    // or_insert_with_key — closure receives the key.
    map.entry("length_of_key")
        .or_insert_with_key(|k| k.len() as u32);
    println!("  length_of_key: {}", map[&"length_of_key"]); // 13

    // or_default — inserts V::default() if absent.
    map.entry("zero").or_default();
    println!("  zero: {}", map[&"zero"]); // 0

    // OccupiedEntry gives direct mutable access.
    if let anchormap::Entry::Occupied(mut e) = map.entry("hits") {
        let old = e.insert(100);
        println!("  replaced hits {old} → {}", e.get());
    }
}

// ── 7. Exclusive mutation (&mut self) ─────────────────────────────────────────

fn exclusive_mutation() {
    println!("\n── 7. Exclusive mutation ──");

    let mut map = HashMap::<u32, u32>::new(64);
    for i in 0..10u32 { map.insert(i, i); }

    // get_mut — direct mutable reference.
    if let Some(v) = map.get_mut(&5) {
        *v *= 100;
    }
    println!("  key 5 after get_mut: {}", map[&5]); // 500

    // modify — apply a closure to an existing value.
    map.modify(&3, |v| *v += 1_000);
    println!("  key 3 after modify: {}", map[&3]); // 1003

    // iter_mut — mutate all values in place.
    for (_, v) in map.iter_mut() { *v += 1; }
    println!("  key 0 after iter_mut +1: {}", map[&0]); // 1

    // values_mut — same, values only.
    for v in map.values_mut() { *v = v.saturating_sub(1); }

    // retain — remove entries matching a predicate (odd keys here).
    let before = map.len();
    map.retain(|k, _| k % 2 == 0);
    println!("  retain(even keys): {before} → {} entries", map.len());

    // remove_entry — removes and returns the (K, V) pair.
    let removed = map.remove_entry(&0);
    println!("  remove_entry(0): {removed:?}");

    // clear — drops all entries, keeps allocated memory.
    map.clear();
    println!("  after clear: len={}, capacity={}", map.len(), map.capacity());
}

// ── 8. Capacity management ────────────────────────────────────────────────────

fn capacity_management() {
    println!("\n── 8. Capacity management ──");

    // reserve — pre-allocate on &self; usable concurrently.
    let map = HashMap::<u32, u32>::new(16);
    map.reserve(10_000);
    println!("  after reserve(10_000): capacity={}", map.capacity());

    // Insert enough entries to force multiple segments, then compact.
    // new(1) starts with 16 slots; 500 entries force ~6 segments of
    // geometrically increasing size (16+32+64+128+256+512 = 1008 slots).
    let mut map = HashMap::<u32, u32>::new(1);
    for i in 0..500u32 { map.insert(i, i); }
    println!("  500 inserts into new(1): capacity={}", map.capacity());

    // shrink_to_fit — collect all entries, free all segments, re-insert into
    // a single right-sized segment.  Capacity may increase slightly if the
    // sum of small segments was less than a single power-of-two segment needs.
    map.shrink_to_fit();
    println!("  after shrink_to_fit:     capacity={}, len={}", map.capacity(), map.len());
    assert_eq!(map.len(), 500);
}

// ── 9. Iteration and drain ────────────────────────────────────────────────────

fn iteration_and_drain() {
    println!("\n── 9. Iteration and drain ──");

    let mut map = HashMap::<u32, u32>::new(64);
    for i in 0..5u32 { map.insert(i, i * i); }

    // iter — lock-free snapshot of (&K, &V) pairs.
    let sum: u32 = map.values().sum();
    println!("  sum of squares 0..5: {sum}"); // 0+1+4+9+16 = 30

    // keys / values
    let mut keys: Vec<u32> = map.keys().copied().collect();
    keys.sort_unstable();
    println!("  keys: {keys:?}");

    // for-loop via IntoIterator for &HashMap
    let mut count = 0usize;
    for (_k, _v) in &map { count += 1; }
    println!("  iterated {count} entries via for-loop");

    // drain — consumes all entries, leaves map empty.
    let pairs: Vec<(u32, u32)> = map.drain().collect();
    println!("  drain yielded {} pairs, map empty: {}", pairs.len(), map.is_empty());

    // into_iter — consuming iteration.
    let map2: HashMap<u32, u32> = (0..4u32).map(|i| (i, i * 3)).collect();
    let mut vals: Vec<u32> = map2.into_values().collect();
    vals.sort_unstable();
    println!("  into_values: {vals:?}");
}

// ── 10. Collection traits ─────────────────────────────────────────────────────

fn collection_traits() {
    println!("\n── 10. Collection traits ──");

    // FromIterator — collect a (K, V) iterator into a map.
    let map: HashMap<&str, usize> = ["alpha", "beta", "gamma"]
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, i))
        .collect();
    println!("  from_iter: {:?}", map);

    // Extend — add entries from an iterator (first-wins on duplicates).
    let mut map: HashMap<u32, u32> = (0..3).map(|i| (i, i)).collect();
    map.extend((2..5).map(|i| (i, i * 100)));
    let mut entries: Vec<(u32, u32)> = map.iter().map(|(&k, &v)| (k, v)).collect();
    entries.sort_unstable();
    println!("  after extend (first-wins): {entries:?}");

    // Index — panics if key absent, useful when presence is guaranteed.
    let map: HashMap<&str, u32> = [("x", 10), ("y", 20)].iter().copied().collect();
    println!("  map[\"x\"] = {}", map[&"x"]);

    // Clone — deep copy with all entries.
    let clone = map.clone();
    println!("  clone == original: {}", map == clone);
}

// ── 11. Custom hasher ─────────────────────────────────────────────────────────

fn custom_hasher() {
    println!("\n── 11. Custom hasher ──");

    // Default: AHash (fast, not DoS-resistant).
    let fast_map = HashMap::<String, u32>::new(64);
    fast_map.insert("hello".to_string(), 1);

    // StandardHasher: SipHash 1-3, the same as std::collections::HashMap.
    // Use this when keys come from untrusted input (e.g. HTTP headers, user IDs).
    let safe_map = HashMap::<String, u32, StandardHasher>::with_hasher(
        64,
        StandardHasher::new(),
    );
    safe_map.insert("hello".to_string(), 1);

    println!("  ahash get:    {:?}", fast_map.get(&"hello".to_string()));
    println!("  siphash get:  {:?}", safe_map.get(&"hello".to_string()));
}
