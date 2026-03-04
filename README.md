# anchormap

[![Crates.io](https://img.shields.io/crates/v/anchormap)](https://crates.io/crates/anchormap)
[![Documentation](https://docs.rs/anchormap/badge.svg)](https://docs.rs/anchormap)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A concurrent hash map where elements are "anchored" and **never move during concurrent access**,
enabling lock-free reads from any number of threads — no locking, no guards, no epoch pins.

## How it works

Conventional open-addressing maps (Robin Hood, Swiss table) move elements during
insertion and resize, making concurrent reads unsafe without locks or guards.
`anchormap` avoids this by keeping element addresses stable under concurrent (`&self`)
access: the map grows by appending new segments (each double the previous size) —
existing segments are never modified or freed while any shared borrow is live. Once a
key is placed in a slot, its address is stable for the lifetime of that shared borrow,
making raw `&V` references safe to hold across concurrent inserts without any guard or
epoch pin.

The probe technique within each segment follows [hashbrown](https://github.com/rust-lang/hashbrown)'s
Swiss-table design: a **flat metadata array** (one byte per slot, 16 slots per cache line)
stores a **7-bit fingerprint** `(hash >> 57) as u8` per occupied slot. A group probe loads
one cache line of metadata, compares fingerprints in bulk, and only dereferences the
data array on a match — roughly 1-in-128 false-positive rate. Groups are visited in a
triangular sequence that covers every group in the segment exactly once.

Writes acquire one of 64 per-stripe locks (indexed by `hash % 64`), each on its own
cache line to eliminate false sharing. Reads are fully lock-free, using Release/Acquire
ordering on the metadata byte to synchronize with writers.

## Quick start

### Installation

```toml
[dependencies]
anchormap = "0.1"

# Enable serde Serialize / Deserialize:
# anchormap = { version = "0.1", features = ["serde"] }
```

### Usage

```rust
use std::sync::Arc;
use anchormap::HashMap;

let map = Arc::new(HashMap::<&str, u32>::new(1024));
map.insert("alpha", 1);
map.insert("beta",  2);

// Any number of threads can read concurrently — no locking required.
std::thread::scope(|s| {
    for _ in 0..8 {
        let map = Arc::clone(&map);
        s.spawn(move || {
            assert_eq!(map.get(&"alpha"), Some(&1));
            assert_eq!(map.get(&"beta"),  Some(&2));
        });
    }
});
```



## API

`HashMap` mirrors `std::collections::HashMap` as closely as its concurrent
nature allows:

- **`&self`** (lock-free): `get`, `get_key_value`, `contains_key`, `iter`,
  `keys`, `values`, `len`, `is_empty`, `capacity`
- **`&self` with internal mutex**: `insert`, `remove`, `get_or_insert`,
  `get_or_insert_with`, `get_or_try_insert_with`, `reserve` — writers are
  serialized, readers are never blocked; the map grows automatically when full
- **`&mut self`** (exclusive): `get_mut`, `iter_mut`, `values_mut`, `modify`,
  `remove_entry`, `clear`, `drain`, `retain`, `entry`, `shrink_to_fit`

`insert` returns `true` if the key was newly inserted, or `false` if it was
already present (the map is unchanged in that case). Use `entry().or_insert()`
for insert-or-update semantics.

All lookup and removal methods accept any borrowed form of the key —
`map.get("str")` works for `HashMap<String, V>` because `String: Borrow<str>`.

Also implements: `Debug`, `Default`, `Clone`, `PartialEq`, `Eq`, `Index<&K>`,
`FromIterator<(K, V)>`, `Extend<(K, V)>`, `IntoIterator`.

### Entry API

```rust
use anchormap::HashMap;

let mut map = HashMap::<&str, u32>::new(64);
*map.entry("hits").or_insert(0) += 1;
map.entry("score").and_modify(|v| *v += 10).or_insert(10);
```

### Concurrent find-or-create

`get_or_insert` and `get_or_insert_with` are available on `&self`, so they
can be called concurrently from multiple threads without `Arc<Mutex<…>>`:

```rust
use std::sync::Arc;
use anchormap::HashMap;

let map = Arc::new(HashMap::<&str, u32>::new(64));

// Returns a reference to the existing value, or inserts 0 and returns that.
let v: &u32 = map.get_or_insert("counter", 0);

// The closure is called only if the key is absent.
let v: &u32 = map.get_or_insert_with("lazy", || expensive_computation());
# fn expensive_computation() -> u32 { 42 }
```

### Custom hasher

The default is **AHash** (same as `hashbrown`). For DoS resistance:

```rust
use anchormap::{HashMap, StandardHasher};

let map = HashMap::<String, u32, StandardHasher>::with_hasher(1024, StandardHasher::new());
```

## Performance characteristics

See [BENCHMARK.md](BENCHMARK.md) for a detailed comparison.

## Important considerations

**Pre-size the map accurately.** `HashMap::new(n)` allocates a single segment
sized for `n` entries at ≤75 % load. Each time the map outgrows all existing
segments a new one is added, double the size of the previous. Those old
segments are never freed, so the total allocated slot capacity can be
significantly larger than the live entry count after several growth events.

In addition to that, inserts iterates all `n` segments, both to check for dupes
and to find the first available slot and scans up to 64 slots per segment, while gets will
iterate segments but end as soon as a match is found.  To guarantee the best performance,
if you know your expected peak size, pass it to `new()` to stay in one segment and
keep utilisation near 75 %.

**Fixed per-instance overhead.** Every `HashMap`, regardless of its capacity,
allocates 64 write-lock stripes each padded to a 64-byte cache line — roughly
4 KB of fixed overhead. This is negligible for large maps but makes `anchormap`
a poor fit for many small, short-lived maps.

**Removal is tombstone-based.** `remove()` marks the slot as a tombstone — the
key and value are *not* dropped immediately. They are freed when the slot is
claimed by a subsequent insert. If you need to reclaim segment memory after bulk removals, call
`shrink_to_fit()` (`&mut self`) — it collects all live entries, frees all
existing segments, allocates a single right-sized segment, and re-inserts.

This is intentional: `get()` returns a plain `&V` reference with no guard, so
the value must remain in place for as long as any caller might hold that
reference. Dropping the value eagerly would dangle those references.


## When to use

`anchormap` excels at concurrent workloads: it scales well with the number of
threads and significantly outperforms every lock-based alternative. It is slower than `papaya` for reads
and consistently underperforms `Dashmap` but it is a good choice when the **reference lifetime** matters more than
raw lookup throughput:

- **Async code** — holding a `DashMap::Ref` across an `.await` point risks a
  deadlock; `&V` from `anchormap` is a plain reference with no such restriction
- **Long-lived lookups** — hand `&V` to callers that hold it for the duration
  of a request, connection, or session, without locking anything for that period
- **Caches and registries** — look up a handler, codec, or config entry once
  and keep the reference for its natural lifetime
- **Zero-copy hot loops** — no guard construction or atomic epoch bump per call;
  reads are pure atomic loads

For comparison `DashMap` returns a `Ref` guard that holds a shard read lock; `papaya` requires
an epoch guard from `map.pin()`. Both tie the reference lifetime to the guard.
In `anchormap`, references are unconditional borrows backed by the guarantee that
values never move during concurrent (`&self`) access.

For workloads where you look up a value, use it immediately, and drop it,
`DashMap` and `papaya` offer higher raw throughput. For maps that will grow
far beyond their initial size, `anchormap` is not a good choice due to its 
memory model and segment allocator. But for maps of a predictable size
involving a workload light on deletes, `anchormap` offers consistent performance
and a more ergonomic and flexible concurrent read API.
