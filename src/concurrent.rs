//! Concurrent HashMap with lock-free reads and automatic growth.
//!
//! # Probe strategy
//!
//! Each segment maintains a flat **metadata array** (one `AtomicU8` per slot)
//! and a flat **data array** (`UnsafeCell<MaybeUninit<(K, V)>>` per slot).
//! Slots are organized into **groups of 16** (one cache line). Probing visits
//! groups in a triangular sequence, enabling compiler auto-vectorization of the
//! group scan into `pcmpeqb`/`pmovmskb` (SSE2) or equivalent.
//!
//! # Metadata byte encoding
//!
//! - `0x80` (`META_EMPTY`): slot never written; data uninitialized.
//! - `0xFE` (`META_TOMBSTONE`): logically deleted; data initialized.
//! - `0xFF` (`META_RESERVED`): claimed by in-flight insert; data not yet
//!   written. Transient — never visible across a completed insert.
//! - `0x00..=0x7F`: **occupied**; high bit clear; lower 7 bits are the
//!   top-7-bit fingerprint `(hash >> 57) as u8`.
//!
//! # Memory model
//!
//! **Segment pointers** (`AtomicPtr`): stored with **Release**, loaded with
//! **Acquire**. A non-null pointer means the segment is fully initialized.
//!
//! **Slot metadata** (`AtomicU8`): writers CAS `META_EMPTY`/`META_TOMBSTONE`
//! → `META_RESERVED` to claim a slot, write `(K, V)`, then **Release**-store
//! the fingerprint. Readers **Acquire**-load the metadata byte and only
//! dereference data when a fingerprint match is confirmed. The Release-Acquire
//! pair establishes the happens-before chain for data visibility.
//!
//! `META_TOMBSTONE` is used by [`HashMap::remove`] for logical deletion without
//! invalidating outstanding `&V` references from [`HashMap::get`].
//!
//! # Locking
//!
//! Writes (`insert`, `remove`, `modify`) are serialized per stripe: there are
//! [`NUM_STRIPES`] independent write locks, indexed by `hash % NUM_STRIPES`.
//! Concurrent writers on different keys that fall in different stripes run in
//! parallel; slot conflicts are resolved by CAS. A separate `growth_lock`
//! serializes segment allocation.

use std::borrow::Borrow;
use std::cell::UnsafeCell;
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicIsize, AtomicPtr, AtomicU64, AtomicU8, AtomicUsize, Ordering};

use parking_lot::Mutex;

use crate::hashing::{AHashHasher, TableHasher};

/// Number of independent write-lock stripes.
///
/// Writes for key with hash `h` acquire `write_stripes[h % NUM_STRIPES].lock`.
/// Same-stripe writes are serialized; different-stripe writes run in parallel.
const NUM_STRIPES: usize = 64;

// ── Probe-engine constants ────────────────────────────────────────────────────

/// Number of slots per probe group. 16 bytes fits in one SSE2/NEON register.
const GROUP_SIZE: usize = 16;

/// Maximum groups probed per segment per operation.
/// Total slots scanned per segment = 4 × 16 = 64.
///
/// At 50 % load, the probability that all 64 probe positions are occupied is
/// (0.5)^64 ≈ 5×10⁻²⁰ — effectively zero. Keys that hash to a full probe
/// region spill to the next segment (which also uses this limit), so
/// correctness is maintained regardless of load factor.
const GROUP_PROBE_LIMIT: usize = 4;

/// Slot never written; data uninitialized.
const META_EMPTY: u8 = 0x80;
/// Logically deleted; data is initialized and must be dropped on reuse/drop.
const META_TOMBSTONE: u8 = 0xFE;
/// Claimed by an in-flight insert; data not yet written. Transient state.
const META_RESERVED: u8 = 0xFF;
// Occupied slots hold a 7-bit fingerprint in 0x00..=0x7F (high bit = 0).

/// Maximum number of growth segments.
///
/// 32 segments starting at 16 slots and doubling gives capacity for
/// 16 × (2³² − 1) ≈ 68 billion entries before `TableFull` is returned.
pub const MAX_SEGMENTS: usize = 32;

// ── Write lane ───────────────────────────────────────────────────────────────

/// One write lane: a stripe lock paired with its element count.
///
/// Co-locating `count` with `lock` means the count update is free — when a
/// writer holds the stripe lock that cache line is already in its L1 cache,
/// so the increment/decrement costs no extra cache miss.
///
/// Adjacent lanes in the `write_stripes` slice share cache lines (16 bytes
/// per lane → 4 lanes per 64-byte line). The resulting false sharing is mild
/// in practice — writes are short critical sections. True per-lane cache-line
/// isolation (`#[repr(align(64))]`) was benchmarked and found to hurt write
/// throughput: it lets more writers past the stripe locks simultaneously,
/// increasing CAS contention on the metadata array enough to outweigh the
/// false-sharing savings.
struct WriteLane {
    lock: Mutex<()>,
    count: AtomicIsize,
}

// ── Internal type aliases ─────────────────────────────────────────────────────

type DataCell<K, V> = UnsafeCell<MaybeUninit<(K, V)>>;

// ── Bloom filter ──────────────────────────────────────────────────────────────

/// A 512-bit Bloom filter (8 × `AtomicU64`) with k=2 hash functions.
///
/// Used as a per-segment membership filter: `maybe_contains` returns `false`
/// only when the key is *definitely* absent (false negatives are impossible).
/// False positives (~0.2–2.7 % for small segments) cause an unnecessary probe.
///
/// Bit positions are derived from non-overlapping ranges of the hash, separate
/// from the 7-bit metadata fingerprint (bits 57–63):
///   p1 = hash as usize & 511         (bits 0–8)
///   p2 = (hash >> 20) as usize & 511 (bits 20–28)
///
/// The filter is monotonically append-only: bits are only ever set, never
/// cleared. Ordering: `set` uses `fetch_or(Release)`; `maybe_contains` uses
/// `load(Acquire)`.
struct BloomFilter {
    bits: [AtomicU64; 8],
    /// When `false` (segment 0): `set` is a no-op and `maybe_contains` always
    /// returns `true`. Segment 0 is always probed via the `probe_seg0` fast
    /// path, so its bloom bits are never consulted — maintaining them would
    /// add two atomic RMW operations per insert with zero benefit.
    active: bool,
}

impl BloomFilter {
    fn new(active: bool) -> Self {
        Self { bits: std::array::from_fn(|_| AtomicU64::new(0)), active }
    }

    /// Set the two bit positions for `hash`. No-op when inactive (segment 0).
    #[inline]
    fn set(&self, hash: u64) {
        if !self.active { return; }
        let p1 = hash as usize & 511;
        let p2 = (hash >> 20) as usize & 511;
        self.bits[p1 / 64].fetch_or(1u64 << (p1 % 64), Ordering::Release);
        self.bits[p2 / 64].fetch_or(1u64 << (p2 % 64), Ordering::Release);
    }

    /// Returns `true` if the key *might* be present; `false` means definitely absent.
    /// Always returns `true` when inactive (segment 0), so it is never skipped.
    #[inline]
    fn maybe_contains(&self, hash: u64) -> bool {
        if !self.active { return true; }
        let p1 = hash as usize & 511;
        let p2 = (hash >> 20) as usize & 511;
        let w1 = self.bits[p1 / 64].load(Ordering::Acquire);
        let w2 = self.bits[p2 / 64].load(Ordering::Acquire);
        (w1 >> (p1 % 64)) & 1 == 1 && (w2 >> (p2 % 64)) & 1 == 1
    }
}

// ── Segment ───────────────────────────────────────────────────────────────────

/// An independent flat allocation with a separate metadata and data array.
/// Segments are never freed or moved while the map is alive.
struct Segment<K, V> {
    /// One metadata byte per slot.  Groups of `GROUP_SIZE` consecutive bytes
    /// form one probe group (≈ one cache line).
    meta: Box<[AtomicU8]>,
    /// One data cell per slot, indexed identically to `meta`.
    data: Box<[DataCell<K, V>]>,
    /// Total number of slots (`num_groups * GROUP_SIZE`).
    size: usize,
    /// `size / GROUP_SIZE`.  Always a power of two.
    num_groups: usize,
    /// `num_groups - 1`.  Used for masking group indices.
    group_mask: usize,
    /// Per-segment Bloom filter for fast skip on miss path.
    bloom: BloomFilter,
}

// SAFETY: all mutable access is synchronized via atomics, stripe locks, or
// exclusive &mut self; data is only read after confirming metadata state.
unsafe impl<K: Send + Sync, V: Send + Sync> Send for Segment<K, V> {}
unsafe impl<K: Send + Sync, V: Send + Sync> Sync for Segment<K, V> {}

impl<K, V> Segment<K, V> {
    /// Allocate a new segment with `size` slots (all `META_EMPTY`).
    ///
    /// `size` must be a multiple of `GROUP_SIZE` and a power of two.
    fn new(size: usize, active_bloom: bool) -> Self {
        debug_assert!(size >= GROUP_SIZE, "size must be ≥ GROUP_SIZE");
        debug_assert!(size.is_multiple_of(GROUP_SIZE), "size must be a multiple of GROUP_SIZE");
        debug_assert!(size.is_power_of_two(), "size must be a power of two");

        let meta = (0..size)
            .map(|_| AtomicU8::new(META_EMPTY))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let data = (0..size)
            .map(|_| DataCell::new(MaybeUninit::uninit()))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let num_groups = size / GROUP_SIZE;
        let group_mask = num_groups - 1;
        Self { meta, data, size, num_groups, group_mask, bloom: BloomFilter::new(active_bloom) }
    }

    /// Lock-free probe: returns `(slot_idx, &K, &V)` if `key` is present.
    ///
    /// Shared implementation for [`get_pair`], [`get`], and [`get_slot_idx`].
    /// The Relaxed→Acquire double-load amortizes the fence cost: we only pay
    /// for Acquire on a fingerprint match (1-in-128 probability per slot).
    #[inline]
    fn find_occupied<Q>(&self, hash: u64, key: &Q) -> Option<(usize, &K, &V)>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let fp = (hash >> 57) as u8; // 7-bit fingerprint; always in 0x00..=0x7F
        let start = (hash as usize) & self.group_mask;
        let limit = GROUP_PROBE_LIMIT.min(self.num_groups);

        let mut probe_offset = 0; // This tracks the triangular number T_j
        for j in 0..limit {
            let g = (start + probe_offset) & self.group_mask;
            let base = g * GROUP_SIZE;
            for i in 0..GROUP_SIZE {
                let slot = base + i;
                if self.meta[slot].load(Ordering::Relaxed) == fp
                    && self.meta[slot].load(Ordering::Acquire) == fp
                {
                    // SAFETY: fp ∈ 0x00..=0x7F ⟹ OCCUPIED; Release/Acquire
                    // chain ensures the (K, V) write in write_slot is visible.
                    let (k, v) =
                        unsafe { (*self.data[slot].get()).assume_init_ref() };
                    if k.borrow() == key {
                        return Some((slot, k, v));
                    }
                }
            }
            probe_offset += j + 1; // Prepare the next triangular increment
        }
        None
    }

    /// Lock-free probe. Returns `Some((&K, &V))` if `key` is present.
    #[inline]
    fn get_pair<Q>(&self, hash: u64, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.find_occupied(hash, key).map(|(_, k, v)| (k, v))
    }

    /// Lock-free probe. Returns `Some(&V)` if `key` is present.
    #[inline]
    fn get<Q>(&self, hash: u64, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.find_occupied(hash, key).map(|(_, _, v)| v)
    }

    /// Lock-free slot lookup. Returns the slot index of `key` if present.
    ///
    /// Used by `remove` to locate the slot before acquiring the stripe lock,
    /// so the locked phase is an O(1) verify + store.
    #[inline]
    fn get_slot_idx<Q>(&self, hash: u64, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.find_occupied(hash, key).map(|(slot, _, _)| slot)
    }

    /// Finds the first `META_EMPTY` or `META_TOMBSTONE` slot in the probe
    /// sequence for `hash`. Used by `insert` after the dup check has already
    /// been done externally via `get`.
    ///
    /// No key comparisons — scans metadata only (one byte per slot).
    fn find_available(&self, hash: u64) -> Option<usize> {
        let start = (hash as usize) & self.group_mask;
        let limit = GROUP_PROBE_LIMIT.min(self.num_groups);

        let mut probe_offset = 0; // This tracks the triangular number T_j
        for j in 0..limit {
            let g = (start + probe_offset) & self.group_mask;
            let base = g * GROUP_SIZE;
            for i in 0..GROUP_SIZE {
                let slot = base + i;
                let state = self.meta[slot].load(Ordering::Relaxed);
                if state == META_EMPTY || state == META_TOMBSTONE {
                    return Some(slot);
                }
            }
            probe_offset += j + 1; // Prepare the next triangular increment
        }
        None
    }

    /// Atomically claims the first `META_EMPTY` or `META_TOMBSTONE` slot via
    /// CAS (`→ META_RESERVED`).
    ///
    /// Returns `Some((slot_idx, old_state))` on success, `None` if the probe
    /// sequence is exhausted. If `old_state == META_TOMBSTONE`, the caller must
    /// drop the slot's data before writing new data.
    fn try_claim_slot(&self, hash: u64) -> Option<(usize, u8)> {
        let start = (hash as usize) & self.group_mask;
        let limit = GROUP_PROBE_LIMIT.min(self.num_groups);

        let mut probe_offset = 0; // This tracks the triangular number T_j
        for j in 0..limit {
            let g = (start + probe_offset) & self.group_mask;
            let base = g * GROUP_SIZE;
            for i in 0..GROUP_SIZE {
                let slot = base + i;
                let state = self.meta[slot].load(Ordering::Relaxed);
                if (state == META_EMPTY || state == META_TOMBSTONE)
                    && self
                        .meta[slot]
                        .compare_exchange(
                            state,
                            META_RESERVED,
                            Ordering::AcqRel,
                            Ordering::Relaxed,
                        )
                        .is_ok()
                {
                    return Some((slot, state));
                    // On CAS failure a concurrent writer claimed it; continue scanning.
                }
            }
            probe_offset += j + 1; // Prepare the next triangular increment
        }
        None
    }

    /// Combined probe for entry/remove/modify operations.
    ///
    /// Returns:
    /// - `Ok(slot_idx)` if `key` was found in an OCCUPIED slot.
    /// - `Err(Some(slot_idx))` if key is absent; `slot_idx` is the first
    ///   `META_EMPTY` slot.
    /// - `Err(None)` if key is absent and no empty slot was found.
    ///
    /// `META_TOMBSTONE` slots are logically absent and are skipped (not
    /// handed out as empty; insert reuses them via `find_available`).
    ///
    /// Uses `Relaxed` ordering: safe under a stripe lock or `&mut self`.
    fn probe_entry<Q>(&self, hash: u64, key: &Q) -> Result<usize, Option<usize>>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let fp = (hash >> 57) as u8;
        let start = (hash as usize) & self.group_mask;
        let limit = GROUP_PROBE_LIMIT.min(self.num_groups);
        let mut first_empty: Option<usize> = None;

        let mut probe_offset = 0; // This tracks the triangular number T_j
        for j in 0..limit {
            let g = (start + probe_offset) & self.group_mask;
            let base = g * GROUP_SIZE;
            for i in 0..GROUP_SIZE {
                let slot = base + i;
                match self.meta[slot].load(Ordering::Relaxed) {
                    META_EMPTY => {
                        if first_empty.is_none() {
                            first_empty = Some(slot);
                        }
                    }
                    META_TOMBSTONE | META_RESERVED => {} // skip
                    b if b == fp => {
                        // SAFETY: OCCUPIED + stripe lock or &mut self ⟹ stable.
                        let (k, _) =
                            unsafe { (*self.data[slot].get()).assume_init_ref() };
                        if k.borrow() == key {
                            return Ok(slot);
                        }
                    }
                    _ => {} // different fingerprint
                }
            }
            probe_offset += j + 1;
        }
        Err(first_empty)
    }

    /// Combined duplicate-check and available-slot probe for insert operations.
    ///
    /// Scans the probe sequence **once** and simultaneously:
    /// - Records the first `META_EMPTY` or `META_TOMBSTONE` slot as the write
    ///   candidate (`first_available`).
    /// - Checks every fingerprint-matched OCCUPIED slot for a duplicate key.
    ///
    /// Returns `(first_available, dup_value)`:
    /// - `first_available`: `Some((slot, old_state))` if a reusable slot was found.
    /// - `dup_value`: `Some(&V)` if the key already exists; `None` if absent.
    ///
    /// SAFETY: caller must hold the stripe lock for `hash % NUM_STRIPES`.
    /// Relaxed ordering is used throughout (the lock provides the happens-before
    /// edge for the dup check; the CAS in the caller provides it for the write).
    fn probe_for_insert<'s>(&'s self, hash: u64, key: &K) -> (Option<(usize, u8)>, Option<&'s V>)
    where
        K: Eq,
    {
        let fp = (hash >> 57) as u8;
        let start = (hash as usize) & self.group_mask;
        let limit = GROUP_PROBE_LIMIT.min(self.num_groups);
        let mut first_available: Option<(usize, u8)> = None;

        let mut probe_offset = 0;
        for j in 0..limit {
            let g = (start + probe_offset) & self.group_mask;
            let base = g * GROUP_SIZE;
            for i in 0..GROUP_SIZE {
                let slot = base + i;
                match self.meta[slot].load(Ordering::Relaxed) {
                    META_EMPTY => {
                        if first_available.is_none() {
                            first_available = Some((slot, META_EMPTY));
                        }
                    }
                    META_TOMBSTONE => {
                        if first_available.is_none() {
                            first_available = Some((slot, META_TOMBSTONE));
                        }
                    }
                    META_RESERVED => {} // transient; skip
                    b if b == fp => {
                        // SAFETY: OCCUPIED + stripe lock ⟹ data stable.
                        let (k, v) = unsafe { (*self.data[slot].get()).assume_init_ref() };
                        if k == key {
                            return (first_available, Some(v));
                        }
                    }
                    _ => {} // different fingerprint
                }
            }
            probe_offset += j + 1;
        }
        (first_available, None)
    }

    /// Write `(key, value)` to `slot` and publish the fingerprint with Release.
    ///
    /// SAFETY: `slot` must be `META_RESERVED` (claimed by CAS). If reusing a
    /// tombstone the caller must have dropped the old data before calling this.
    unsafe fn write_slot(&self, slot: usize, hash: u64, key: K, value: V) {
        (*self.data[slot].get()).write((key, value));
        // Release: ensures the bloom bit is visible to any reader who later
        // observes the metadata fingerprint via Acquire.
        self.bloom.set(hash);
        // Release: pairs with readers' Acquire load of the metadata byte.
        self.meta[slot].store((hash >> 57) as u8, Ordering::Release);
    }
}

impl<K, V> Drop for Segment<K, V> {
    fn drop(&mut self) {
        for (i, meta) in self.meta.iter_mut().enumerate() {
            let b = *meta.get_mut();
            // META_EMPTY (0x80) and META_RESERVED (0xFF) never hold init data.
            // Everything else — OCCUPIED (0x00..=0x7F) and META_TOMBSTONE (0xFE)
            // — holds initialized (K, V) that must be dropped.
            if b != META_EMPTY && b != META_RESERVED {
                // SAFETY: OCCUPIED and TOMBSTONE slots hold initialized data.
                unsafe { (*self.data[i].get()).assume_init_drop() };
            }
        }
    }
}

// ── Iterator types ────────────────────────────────────────────────────────────

/// Lock-free snapshot iterator over `(&K, &V)` pairs.
///
/// Yielded references are valid for the lifetime of the map. Elements inserted
/// after the iterator is created may or may not appear; no element appears twice.
pub struct Iter<'a, K, V, H = AHashHasher> {
    map: &'a HashMap<K, V, H>,
    seg_idx: usize,
    slot_idx: usize,
}

impl<'a, K, V, H> Iterator for Iter<'a, K, V, H> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.seg_idx >= MAX_SEGMENTS {
                return None;
            }
            let ptr = self.map.segments[self.seg_idx].load(Ordering::Acquire);
            if ptr.is_null() {
                return None;
            }
            // SAFETY: non-null ptr; segment lives for 'a.
            let seg: &'a Segment<K, V> = unsafe { &*ptr };

            if self.slot_idx >= seg.size {
                self.seg_idx += 1;
                self.slot_idx = 0;
                continue;
            }

            let idx = self.slot_idx;
            self.slot_idx += 1;

            // Occupied: high bit clear (fingerprint byte in 0x00..=0x7F).
            let b = seg.meta[idx].load(Ordering::Acquire);
            if b & 0x80 == 0 {
                // SAFETY: OCCUPIED + Acquire pairs with Release in write_slot.
                let pair = unsafe { (*seg.data[idx].get()).assume_init_ref() };
                return Some((&pair.0, &pair.1));
            }
        }
    }
}

/// Consuming iterator over `(K, V)` pairs.
pub struct IntoIter<K, V, H = AHashHasher> {
    map: HashMap<K, V, H>,
    seg_idx: usize,
    slot_idx: usize,
}

impl<K, V, H> Iterator for IntoIter<K, V, H> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.seg_idx >= MAX_SEGMENTS {
                return None;
            }
            let ptr = self.map.segments[self.seg_idx].load(Ordering::Relaxed);
            if ptr.is_null() {
                return None;
            }
            let seg: &Segment<K, V> = unsafe { &*ptr };

            if self.slot_idx >= seg.size {
                self.seg_idx += 1;
                self.slot_idx = 0;
                continue;
            }

            let idx = self.slot_idx;
            self.slot_idx += 1;

            if seg.meta[idx].load(Ordering::Relaxed) & 0x80 == 0 {
                // SAFETY: OCCUPIED ⟹ data initialized; assume_init_read() moves
                // the value out; we mark META_EMPTY so Segment::drop skips it.
                let kv = unsafe { (*seg.data[idx].get()).assume_init_read() };
                seg.meta[idx].store(META_EMPTY, Ordering::Relaxed);
                return Some(kv);
            }
        }
    }
}

/// Draining iterator over `(K, V)` pairs, obtained via [`HashMap::drain`].
pub struct Drain<'a, K, V, H = AHashHasher> {
    map: &'a mut HashMap<K, V, H>,
    seg_idx: usize,
    slot_idx: usize,
}

impl<'a, K, V, H> Iterator for Drain<'a, K, V, H> {
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        loop {
            if self.seg_idx >= MAX_SEGMENTS {
                return None;
            }
            let ptr = self.map.segments[self.seg_idx].load(Ordering::Relaxed);
            if ptr.is_null() {
                return None;
            }
            let seg: &Segment<K, V> = unsafe { &*ptr };

            if self.slot_idx >= seg.size {
                self.seg_idx += 1;
                self.slot_idx = 0;
                continue;
            }

            let idx = self.slot_idx;
            self.slot_idx += 1;

            if seg.meta[idx].load(Ordering::Relaxed) & 0x80 == 0 {
                // SAFETY: OCCUPIED ⟹ initialized; assume_init_read() moves the
                // pair out, then we mark META_EMPTY so Segment::drop skips it.
                let kv = unsafe { (*seg.data[idx].get()).assume_init_read() };
                seg.meta[idx].store(META_EMPTY, Ordering::Relaxed);
                return Some(kv);
            }
        }
    }
}

impl<'a, K, V, H> Drop for Drain<'a, K, V, H> {
    fn drop(&mut self) {
        // Exhaust remaining occupied slots, dropping their data in place.
        // (Cannot call self.by_ref() here as that requires the K:Hash+H:TableHasher
        // bounds that Drop impls cannot add beyond the struct definition.)
        loop {
            if self.seg_idx >= MAX_SEGMENTS { break; }
            let ptr = self.map.segments[self.seg_idx].load(Ordering::Relaxed);
            if ptr.is_null() { break; }
            let seg: &Segment<K, V> = unsafe { &*ptr };
            if self.slot_idx >= seg.size {
                self.seg_idx += 1;
                self.slot_idx = 0;
                continue;
            }
            let idx = self.slot_idx;
            self.slot_idx += 1;
            if seg.meta[idx].load(Ordering::Relaxed) & 0x80 == 0 {
                unsafe { (*seg.data[idx].get()).assume_init_drop() };
                seg.meta[idx].store(META_EMPTY, Ordering::Relaxed);
            }
        }
        // After exhausting all elements the map is empty; reset all stripe counts.
        for lane in self.map.write_stripes.iter() {
            lane.count.store(0, Ordering::Relaxed);
        }
    }
}

// ── Entry API ─────────────────────────────────────────────────────────────────

/// A view into a single entry in the map, obtained via [`HashMap::entry`].
pub enum Entry<'a, K, V, H = AHashHasher> {
    /// An occupied entry.
    Occupied(OccupiedEntry<'a, K, V, H>),
    /// A vacant entry.
    Vacant(VacantEntry<'a, K, V, H>),
}

/// An occupied entry in [`HashMap`].
pub struct OccupiedEntry<'a, K, V, H = AHashHasher> {
    /// Holds the exclusive borrow of the map for lifetime `'a`; not accessed directly.
    _map: &'a mut HashMap<K, V, H>,
    /// Raw pointer to the segment containing the occupied slot.
    seg_ptr: *const Segment<K, V>,
    /// Index of the occupied slot within the segment's arrays.
    slot_idx: usize,
    /// Hash of the entry's key; used to identify the stripe for count updates.
    hash: u64,
}

/// A vacant entry in [`HashMap`].
pub struct VacantEntry<'a, K, V, H = AHashHasher> {
    map: &'a mut HashMap<K, V, H>,
    key: K,
    /// Hash of `key`, used to compute the fingerprint when inserting.
    hash: u64,
    /// First empty slot found during the entry scan, if any.
    /// `None` means all probe sequences exhausted; insert will grow the map.
    slot: Option<(*const Segment<K, V>, usize)>,
    _phantom: PhantomData<H>,
}

impl<'a, K, V, H> OccupiedEntry<'a, K, V, H> {
    #[inline]
    fn seg(&self) -> &Segment<K, V> {
        // SAFETY: seg_ptr came from a live segment; &mut self prevents
        // concurrent access while this entry is held.
        unsafe { &*self.seg_ptr }
    }

    /// Returns a reference to the key of this entry.
    pub fn key(&self) -> &K {
        // SAFETY: slot is OCCUPIED; data initialized.
        unsafe { &(*self.seg().data[self.slot_idx].get()).assume_init_ref().0 }
    }

    /// Returns a reference to the value of this entry.
    pub fn get(&self) -> &V {
        unsafe { &(*self.seg().data[self.slot_idx].get()).assume_init_ref().1 }
    }

    /// Returns a mutable reference to the value of this entry.
    pub fn get_mut(&mut self) -> &mut V {
        unsafe { &mut (*self.seg().data[self.slot_idx].get()).assume_init_mut().1 }
    }

    /// Converts into a mutable reference valid for the map's borrow lifetime.
    pub fn into_mut(self) -> &'a mut V {
        unsafe { &mut (*(*self.seg_ptr).data[self.slot_idx].get()).assume_init_mut().1 }
    }

    /// Sets the value of the entry, returning the old value.
    pub fn insert(&mut self, value: V) -> V {
        unsafe {
            let pair: &mut (K, V) =
                (*self.seg().data[self.slot_idx].get()).assume_init_mut();
            std::mem::replace(&mut pair.1, value)
        }
    }

    /// Removes the entry, returning the value.
    pub fn remove(self) -> V {
        self.remove_entry().1
    }

    /// Removes the entry, returning the key-value pair.
    pub fn remove_entry(self) -> (K, V) {
        // SAFETY: OCCUPIED ⟹ data initialized. assume_init_read() moves the
        // pair out; we mark META_EMPTY so Segment::drop skips it.
        let kv = unsafe {
            (*(*self.seg_ptr).data[self.slot_idx].get()).assume_init_read()
        };
        // Relaxed: &mut self excludes concurrent access.
        unsafe {
            (*self.seg_ptr).meta[self.slot_idx].store(META_EMPTY, Ordering::Relaxed)
        };
        let stripe_idx = self.hash as usize % NUM_STRIPES;
        self._map.write_stripes[stripe_idx].count.fetch_sub(1, Ordering::Relaxed);
        kv
    }
}

impl<'a, K, V, H> VacantEntry<'a, K, V, H>
where
    K: Eq + Hash,
    H: TableHasher,
{
    /// Returns a reference to the key that would be used when inserting.
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Consumes the entry, returning the key.
    pub fn into_key(self) -> K {
        self.key
    }

    /// Inserts the value into the entry and returns a mutable reference to it.
    pub fn insert(self, value: V) -> &'a mut V {
        if let Some((seg_ptr, slot_idx)) = self.slot {
            // Write to the pre-found empty slot.
            // SAFETY: slot was found META_EMPTY during the scan; &mut self
            // prevents concurrent access.
            unsafe {
                let seg = &*seg_ptr;
                (*seg.data[slot_idx].get()).write((self.key, value));
                seg.bloom.set(self.hash);
                seg.meta[slot_idx]
                    .store((self.hash >> 57) as u8, Ordering::Release);
            }
            self.map.write_stripes[self.hash as usize % NUM_STRIPES]
                .count.fetch_add(1, Ordering::Relaxed);
            unsafe { &mut (*(*seg_ptr).data[slot_idx].get()).assume_init_mut().1 }
        } else {
            // All probe sequences exhausted — grow via insert_locked.
            let val_ptr = self.map.insert_locked(self.key, value);
            unsafe { &mut *val_ptr }
        }
    }
}

impl<'a, K, V, H> Entry<'a, K, V, H>
where
    K: Eq + Hash,
    H: TableHasher,
{
    /// Returns a reference to the key of this entry.
    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(e) => e.key(),
            Entry::Vacant(e) => e.key(),
        }
    }

    /// Ensures a value is in the entry, inserting `default` if vacant.
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(default),
        }
    }

    /// Ensures a value is in the entry, calling `f` to produce one if vacant.
    pub fn or_insert_with(self, f: impl FnOnce() -> V) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(f()),
        }
    }

    /// Like `or_insert_with` but the closure receives the entry's key.
    pub fn or_insert_with_key(self, f: impl FnOnce(&K) -> V) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => {
                let v = f(e.key());
                e.insert(v)
            }
        }
    }

    /// Ensures a value is in the entry using `V::default()` if vacant.
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.or_insert_with(V::default)
    }

    /// Provides in-place mutable access to an occupied entry before potential
    /// inserts. A no-op on a vacant entry.
    pub fn and_modify(self, f: impl FnOnce(&mut V)) -> Self {
        match self {
            Entry::Occupied(mut e) => {
                f(e.get_mut());
                Entry::Occupied(e)
            }
            v => v,
        }
    }
}

// ── HashMap ────────────────────────────────────────────────────────────────────

/// A hash map that allows concurrent lock-free reads and automatically grows.
///
/// Based on flat metadata probing with per-stripe write locks, so `get()` is
/// fully lock-free from any number of threads even while `insert()` is running.
///
/// The API mirrors [`std::collections::HashMap`] as closely as possible:
/// - **`&self`** (concurrent-safe): `get`, `contains_key`, `iter`, `len`, …
/// - **`&mut self`** (exclusive access): `entry`, `drain`, `clear`, …
/// - **`&self` with internal lock**: `insert`, `remove`, `modify`
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use anchormap::HashMap;
///
/// let map = Arc::new(HashMap::<&str, u32>::new(64));
/// map.insert("hello", 42);
///
/// let map2 = Arc::clone(&map);
/// let handle = std::thread::spawn(move || {
///     map2.get(&"hello").copied()
/// });
/// assert_eq!(handle.join().unwrap(), Some(42));
/// ```
pub struct HashMap<K, V, H = AHashHasher> {
    /// Raw pointer into segment 0's metadata array.  Cached inline to avoid
    /// the `Box → AtomicPtr → Segment → Box<meta>` pointer chain on every
    /// lock-free read.  Valid for the entire lifetime of `self` (segment 0 is
    /// never freed while the map is alive).
    seg0_meta: *const AtomicU8,
    /// Raw pointer into segment 0's data array.  Same lifetime guarantee as
    /// `seg0_meta`.
    seg0_data: *const DataCell<K, V>,
    /// Segment 0's `group_mask` (= `num_groups − 1`), used for probe indexing.
    seg0_mask: usize,
    /// Fixed-size array of segment pointers; the first `n_segments` entries
    /// are non-null (stored with Release; loaded with Acquire or Relaxed after
    /// a single Acquire on `n_segments` establishes the happens-before edge).
    segments: Box<[AtomicPtr<Segment<K, V>>; MAX_SEGMENTS]>,
    /// Number of live segments. Release-stored after each allocation; readers
    /// Acquire-load this once to bound their loop.
    n_segments: AtomicUsize,
    total_capacity: AtomicUsize,
    hasher: H,
    /// Per-stripe write lanes. A write for key with hash `h` acquires
    /// `write_stripes[h % NUM_STRIPES].lock`. Same-stripe writes are
    /// serialized; different-stripe writes run in parallel (slot conflicts
    /// resolved by CAS). Each lane's element count is updated while the lock
    /// is held, so the count write costs no additional cache miss.
    write_stripes: Box<[WriteLane]>,
    /// Serializes segment allocation only. Acquired *after* the stripe lock
    /// (never reversed — avoids deadlock).
    growth_lock: Mutex<()>,
}

// SAFETY: see module-level doc for the synchronization argument.
unsafe impl<K, V, H> Sync for HashMap<K, V, H>
where
    K: Send + Sync,
    V: Send + Sync,
    H: Sync,
{}

unsafe impl<K, V, H> Send for HashMap<K, V, H>
where
    K: Send + Sync,
    V: Send + Sync,
    H: Send,
{}

impl<K, V> HashMap<K, V, AHashHasher>
where
    K: Eq + Hash,
{
    /// Creates a new concurrent map sized to hold `capacity` elements.
    ///
    /// The first segment is allocated large enough to hold `capacity` elements
    /// at ≤ 75 % load, rounded up to a power of two (minimum 16 slots).  The
    /// map grows automatically beyond that capacity as needed.
    pub fn new(capacity: usize) -> Self {
        Self::with_hasher(capacity, AHashHasher::default())
    }

    /// Alias for [`new`](Self::new); provided for API parity with
    /// `std::collections::HashMap`.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::new(capacity)
    }
}

impl<K, V, H> HashMap<K, V, H>
where
    K: Eq + Hash,
    H: TableHasher,
{
    /// Creates a new map sized to hold `capacity` elements with a custom hasher.
    pub fn with_capacity_and_hasher(capacity: usize, hasher: H) -> Self {
        Self::with_hasher(capacity, hasher)
    }

    /// Creates a new concurrent map sized to hold `capacity` elements with
    /// the given hasher.
    pub fn with_hasher(capacity: usize, hasher: H) -> Self {
        // Allocate slots for `capacity` elements at ≤ 75 % load factor.
        // slots = ceil(capacity / 0.75) = capacity * 4 / 3 + 1, then rounded
        // up to the next power of two (minimum GROUP_SIZE = 16 slots).
        let initial_size = (capacity.saturating_mul(4) / 3 + 1)
            .max(GROUP_SIZE)
            .next_power_of_two();

        let segments: Box<[AtomicPtr<Segment<K, V>>; MAX_SEGMENTS]> =
            Box::new(std::array::from_fn(|_| AtomicPtr::new(std::ptr::null_mut())));

        let first = Box::into_raw(Box::new(Segment::new(initial_size, false)));
        // Release: segment fully initialized before any reader sees it.
        segments[0].store(first, Ordering::Release);

        // Cache segment 0's raw meta/data pointers inline in the HashMap struct.
        // This eliminates the Box → AtomicPtr → Segment → Box<meta> chain from
        // every lock-free read.  SAFETY: `first` was just constructed and is
        // fully initialized; these pointers are valid for the map's lifetime.
        let seg0_meta = unsafe { (*first).meta.as_ptr() };
        let seg0_data = unsafe { (*first).data.as_ptr() };
        let seg0_mask = unsafe { (*first).group_mask };

        Self {
            seg0_meta,
            seg0_data,
            seg0_mask,
            segments,
            n_segments: AtomicUsize::new(1),
            total_capacity: AtomicUsize::new(initial_size),
            hasher,
            write_stripes: (0..NUM_STRIPES)
                .map(|_| WriteLane {
                    lock: Mutex::new(()),
                    count: AtomicIsize::new(0),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            growth_lock: Mutex::new(()),
        }
    }

    // ── Shared (&self) ────────────────────────────────────────────────────────

    /// Fast-path probe of segment 0 using the inline `seg0_meta`/`seg0_data`
    /// pointers.  Avoids the `Box → AtomicPtr → Segment → Box<meta>` chain
    /// present in a normal segment lookup.
    ///
    /// Returns `Some((slot, &K, &V))` on a hit, `None` on a miss.
    ///
    /// SAFETY: `seg0_meta` and `seg0_data` are valid for `'map`; data is only
    /// dereferenced after an Acquire fingerprint confirmation.
    #[inline]
    fn probe_seg0<'map, Q>(&'map self, hash: u64, key: &Q) -> Option<(usize, &'map K, &'map V)>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let fp = (hash >> 57) as u8;
        let limit = GROUP_PROBE_LIMIT.min(self.seg0_mask + 1);
        let start = (hash as usize) & self.seg0_mask;
        let mut probe_offset = 0;
        for j in 0..limit {
            let g = (start + probe_offset) & self.seg0_mask;
            let base = g * GROUP_SIZE;
            for i in 0..GROUP_SIZE {
                let slot = base + i;
                // SAFETY: seg0_meta points into segment 0's live AtomicU8 array.
                let meta = unsafe { &*self.seg0_meta.add(slot) };
                if meta.load(Ordering::Relaxed) == fp
                    && meta.load(Ordering::Acquire) == fp
                {
                    // SAFETY: seg0_data points into segment 0's data array;
                    // Release/Acquire on the metadata byte ensures visibility.
                    let (k, v) = unsafe {
                        let cell = &*self.seg0_data.add(slot);
                        (*cell.get()).assume_init_ref()
                    };
                    if k.borrow() == key {
                        return Some((slot, k, v));
                    }
                }
            }
            probe_offset += j + 1;
        }
        None
    }

    /// Returns a reference to the value for `key`, or `None` if absent.
    ///
    /// **Lock-free**: safe to call concurrently from any number of threads
    /// while `insert()` / `remove()` are running on other threads.
    ///
    /// Accepts any borrowed form of the key: `map.get("str")` works for
    /// `HashMap<String, V>` because `String: Borrow<str>`.
    #[inline]
    pub fn get<'map, Q>(&'map self, key: &Q) -> Option<&'map V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.hasher.hash_key(key);
        // Fast path: probe segment 0 via inline pointers (no Box indirection).
        if let Some((_, _, v)) = self.probe_seg0(hash, key) {
            return Some(v);
        }
        // Fallback: check segments 1..n for keys that spilled beyond segment 0.
        // One Acquire on n_segments establishes happens-before with all prior
        // segment allocations; subsequent segment pointer loads can be Relaxed.
        let n = self.n_segments.load(Ordering::Acquire);
        for i in 1..n {
            // SAFETY: indices 1..n are non-null; segments live for 'map.
            let seg: &'map Segment<K, V> =
                unsafe { &*self.segments[i].load(Ordering::Relaxed) };
            if seg.bloom.maybe_contains(hash) {
                if let Some(v) = seg.get(hash, key) {
                    return Some(v);
                }
            }
        }
        None
    }

    /// Inserts a key-value pair.
    ///
    /// Returns `true` if the key was newly inserted, or `false` if the key was
    /// already present (the map is left unchanged in that case).
    ///
    /// Use `entry().or_insert()` for insert-or-update semantics, or
    /// `modify()` to mutate a value in place.
    pub fn insert(&self, key: K, value: V) -> bool {
        let hash = self.hasher.hash_key(&key);
        let stripe_idx = hash as usize % NUM_STRIPES;
        let _stripe = self.write_stripes[stripe_idx].lock.lock();

        // Single combined pass across all segments: check for a duplicate AND
        // record the first available slot (META_EMPTY or META_TOMBSTONE).
        //
        // Same key → same stripe; the stripe lock excludes all concurrent
        // inserts/removes of this key, so Relaxed ordering is sufficient for
        // the dup check.  This replaces the former two-pass approach
        // (get_by_hash + try_claim_slot) with one probe sequence.
        let n = self.n_segments.load(Ordering::Acquire);
        let mut claim: Option<(*const Segment<K, V>, usize, u8)> = None;

        for i in 0..n {
            let ptr = self.segments[i].load(Ordering::Relaxed);
            let seg = unsafe { &*ptr };
            if seg.bloom.maybe_contains(hash) {
                // Key might be here: full dup check + available slot probe.
                let (available, dup) = seg.probe_for_insert(hash, &key);
                if dup.is_some() {
                    return false; // key already present
                }
                if claim.is_none() {
                    if let Some((slot, old_state)) = available {
                        claim = Some((ptr as *const _, slot, old_state));
                    }
                }
            } else if claim.is_none() {
                // Key is definitely absent from this segment; skip dup check.
                // Only look for an available slot to use as write target.
                if let Some(slot) = seg.find_available(hash) {
                    let old_state = seg.meta[slot].load(Ordering::Relaxed);
                    claim = Some((ptr as *const _, slot, old_state));
                }
            }
            // If bloom negative and claim already set: skip segment entirely.
        }

        // No duplicate. Claim the identified slot via CAS.
        if let Some((seg_ptr, slot_idx, old_state)) = claim {
            let seg = unsafe { &*seg_ptr };
            if seg.meta[slot_idx]
                .compare_exchange(old_state, META_RESERVED, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                if old_state == META_TOMBSTONE {
                    // SAFETY: TOMBSTONE ⟹ data initialized; drop before overwriting.
                    unsafe { (*seg.data[slot_idx].get()).assume_init_drop() };
                }
                // SAFETY: slot is now RESERVED (claimed by our CAS).
                unsafe { seg.write_slot(slot_idx, hash, key, value) };
                self.write_stripes[stripe_idx].count.fetch_add(1, Ordering::Relaxed);
                return true;
            }
            // CAS lost to a concurrent writer on a different stripe; fall through.
        }

        // No available slot found or CAS failed: full slot search with possible growth.
        self.insert_absent(hash, key, value);
        true
    }

    /// Returns a reference to the value for `key`, inserting `value` if absent.
    ///
    /// Unlike `entry().or_insert()`, this is available on `&self` and is safe
    /// to call concurrently from multiple threads.
    ///
    /// The returned reference is valid for the map's lifetime — no guard is
    /// held after the call returns.
    #[inline]
    pub fn get_or_insert(&self, key: K, value: V) -> &V {
        self.get_or_insert_with(key, || value)
    }

    /// Returns a reference to the value for `key`. If absent, calls `f` to
    /// produce a value, inserts it, and returns a reference.
    ///
    /// `f` is called **only if** the key is not already present. This is
    /// available on `&self` and is safe to call concurrently from multiple
    /// threads.
    ///
    /// The returned reference is valid for the map's lifetime — no guard is
    /// held after the call returns.
    ///
    /// # Warning: closure runs under a write lock
    ///
    /// `f` is called while holding the internal stripe lock for `key`. Keep
    /// closures short and non-blocking: a slow or blocking `f` will stall all
    /// concurrent writes whose keys hash to the same stripe (1 of 64 stripes).
    /// If you need to perform expensive work, compute the value before calling
    /// this method and use [`get_or_insert`](Self::get_or_insert) instead.
    #[inline]
    pub fn get_or_insert_with<'map, F: FnOnce() -> V>(&'map self, key: K, f: F) -> &'map V {
        let hash = self.hasher.hash_key(&key);
        let stripe_idx = hash as usize % NUM_STRIPES;
        let _stripe = self.write_stripes[stripe_idx].lock.lock();

        // Same single-pass combined probe as `insert`: dup check + first-available
        // slot in one scan. Returns &V immediately on dup; otherwise claims and
        // writes on CAS success; falls back to insert_absent on CAS failure or
        // probe-sequence exhaustion.
        let n = self.n_segments.load(Ordering::Acquire);
        let mut claim: Option<(*const Segment<K, V>, usize, u8)> = None;

        for i in 0..n {
            let ptr = self.segments[i].load(Ordering::Relaxed);
            let seg: &'map Segment<K, V> = unsafe { &*ptr };
            if seg.bloom.maybe_contains(hash) {
                // Key might be here: full dup check + available slot probe.
                let (available, dup) = seg.probe_for_insert(hash, &key);
                if let Some(v) = dup {
                    return v;
                }
                if claim.is_none() {
                    if let Some((slot, old_state)) = available {
                        claim = Some((ptr as *const _, slot, old_state));
                    }
                }
            } else if claim.is_none() {
                // Key is definitely absent; skip dup check, only seek a slot.
                if let Some(slot) = seg.find_available(hash) {
                    let old_state = seg.meta[slot].load(Ordering::Relaxed);
                    claim = Some((ptr as *const _, slot, old_state));
                }
            }
            // If bloom negative and claim already set: skip segment entirely.
        }

        if let Some((seg_ptr, slot_idx, old_state)) = claim {
            let seg = unsafe { &*seg_ptr };
            if seg.meta[slot_idx]
                .compare_exchange(old_state, META_RESERVED, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                if old_state == META_TOMBSTONE {
                    unsafe { (*seg.data[slot_idx].get()).assume_init_drop() };
                }
                // SAFETY: slot is RESERVED; write_slot publishes with Release.
                unsafe { seg.write_slot(slot_idx, hash, key, f()) };
                self.write_stripes[stripe_idx].count.fetch_add(1, Ordering::Relaxed);
                // SAFETY: value lives at seg.data[slot_idx] for the map's lifetime.
                return unsafe { &(*seg.data[slot_idx].get()).assume_init_ref().1 };
            }
        }

        // Fallback: full slot search with possible growth.
        // SAFETY: insert_absent places the value in a slot live for the map's lifetime.
        unsafe { &*self.insert_absent(hash, key, f()) }
    }

    /// Core write path: insert `(key, value)` into the first available slot.
    ///
    /// **Preconditions** (callers must uphold):
    /// 1. The stripe lock for `hash % NUM_STRIPES` is held by the caller.
    /// 2. `key` is confirmed absent (dup check already done under the lock).
    ///
    /// Returns a raw pointer to the inserted value, valid for the map's lifetime.
    fn insert_absent(&self, hash: u64, key: K, value: V) -> *const V {
        let stripe_idx = hash as usize % NUM_STRIPES;

        // Fast path: try existing segments. Key confirmed absent so no key
        // comparisons are needed — metadata CAS is sufficient.
        let n = self.n_segments.load(Ordering::Acquire);
        for i in 0..n {
            let ptr = self.segments[i].load(Ordering::Relaxed);
            let seg = unsafe { &*ptr };
            if let Some((slot_idx, old_state)) = seg.try_claim_slot(hash) {
                if old_state == META_TOMBSTONE {
                    // SAFETY: TOMBSTONE ⟹ data initialized; stripe lock
                    // serializes same-stripe writers so pre-tombstone &V
                    // references remain valid until this drop.
                    unsafe { (*seg.data[slot_idx].get()).assume_init_drop() };
                }
                // SAFETY: slot is RESERVED (claimed by us via CAS).
                unsafe { seg.write_slot(slot_idx, hash, key, value) };
                self.write_stripes[stripe_idx].count.fetch_add(1, Ordering::Relaxed);
                return unsafe { &(*seg.data[slot_idx].get()).assume_init_ref().1 };
            }
        }

        // Slow path: all probe sequences exhausted. Acquire growth_lock, check
        // for segments added while we waited, then allocate a new one if needed.
        let _growth = self.growth_lock.lock();

        let mut n_segs = 0usize;
        for i in 0..MAX_SEGMENTS {
            let ptr = self.segments[i].load(Ordering::Acquire);
            if ptr.is_null() {
                break;
            }
            n_segs = i + 1;
            let seg = unsafe { &*ptr };
            if let Some((slot_idx, old_state)) = seg.try_claim_slot(hash) {
                if old_state == META_TOMBSTONE {
                    unsafe { (*seg.data[slot_idx].get()).assume_init_drop() };
                }
                unsafe { seg.write_slot(slot_idx, hash, key, value) };
                self.write_stripes[stripe_idx].count.fetch_add(1, Ordering::Relaxed);
                return unsafe { &(*seg.data[slot_idx].get()).assume_init_ref().1 };
            }
        }

        // Allocate a new segment (double the previous size).
        // With geometric doubling, exhausting all MAX_SEGMENTS segments would
        // require inserting more entries than fit in addressable memory.
        assert!(n_segs < MAX_SEGMENTS, "anchormap: all segments exhausted (unreachable at normal scales)");
        let prev_size = unsafe { (*self.segments[n_segs - 1].load(Ordering::Relaxed)).size };
        let next_size = prev_size.saturating_mul(2).next_power_of_two();
        let new_seg_ptr = Box::into_raw(Box::new(Segment::new(next_size, true)));
        // Release: segment fully initialized before readers can see this ptr.
        self.segments[n_segs].store(new_seg_ptr, Ordering::Release);
        // Release: readers Acquire n_segments; must be after the segment store.
        self.n_segments.fetch_add(1, Ordering::Release);
        self.total_capacity.fetch_add(next_size, Ordering::Relaxed);

        let seg = unsafe { &*new_seg_ptr };
        // New segment is all META_EMPTY — try_claim_slot is guaranteed to succeed.
        let (slot_idx, _) = seg.try_claim_slot(hash)
            .expect("fresh segment must have an empty slot");
        unsafe { seg.write_slot(slot_idx, hash, key, value) };
        self.write_stripes[stripe_idx].count.fetch_add(1, Ordering::Relaxed);
        unsafe { &(*seg.data[slot_idx].get()).assume_init_ref().1 }
    }

    /// Returns `true` if the map contains the given key.
    ///
    /// Accepts any borrowed form of the key: `map.contains_key("str")` works
    /// for `HashMap<String, V>` because `String: Borrow<str>`.
    #[inline]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.get(key).is_some()
    }

    /// Returns the number of elements in the map (approximate under concurrency).
    #[inline]
    pub fn len(&self) -> usize {
        self.write_stripes
            .iter()
            .map(|lane| lane.count.load(Ordering::Relaxed).max(0) as usize)
            .sum()
    }

    /// Returns `true` if the map contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.write_stripes
            .iter()
            .all(|lane| lane.count.load(Ordering::Relaxed) <= 0)
    }

    /// Returns the total allocated slot capacity across all segments.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.total_capacity.load(Ordering::Relaxed)
    }

    /// Returns a lock-free snapshot iterator over all `(&K, &V)` pairs.
    #[inline]
    pub fn iter(&self) -> Iter<'_, K, V, H> {
        Iter { map: self, seg_idx: 0, slot_idx: 0 }
    }

    /// Returns an iterator over all keys.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.iter().map(|(k, _)| k)
    }

    /// Returns an iterator over all values.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.iter().map(|(_, v)| v)
    }

    /// Returns the key-value pair for `key`, or `None` if absent.
    ///
    /// Accepts any borrowed form of the key: `map.get_key_value("str")` works
    /// for `HashMap<String, V>` because `String: Borrow<str>`.
    #[inline]
    pub fn get_key_value<'map, Q>(&'map self, key: &Q) -> Option<(&'map K, &'map V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.hasher.hash_key(key);
        if let Some((_, k, v)) = self.probe_seg0(hash, key) {
            return Some((k, v));
        }
        let n = self.n_segments.load(Ordering::Acquire);
        for i in 1..n {
            let seg: &'map Segment<K, V> =
                unsafe { &*self.segments[i].load(Ordering::Relaxed) };
            if seg.bloom.maybe_contains(hash) {
                if let Some(pair) = seg.get_pair(hash, key) {
                    return Some(pair);
                }
            }
        }
        None
    }

    /// Consumes the map, returning an iterator over all keys.
    pub fn into_keys(self) -> impl Iterator<Item = K> {
        self.into_iter().map(|(k, _)| k)
    }

    /// Consumes the map, returning an iterator over all values.
    pub fn into_values(self) -> impl Iterator<Item = V> {
        self.into_iter().map(|(_, v)| v)
    }

    // ── Shared (&self) with internal lock ─────────────────────────────────────

    /// Removes `key` from the map. Returns `true` if the key was present.
    ///
    /// Accepts any borrowed form of the key: `map.remove("str")` works for
    /// `HashMap<String, V>` because `String: Borrow<str>`.
    ///
    /// Internally marks the slot as `META_TOMBSTONE` rather than freeing it
    /// immediately, so outstanding `&V` references obtained via `get()` remain
    /// valid. The slot data is freed when it is reused by a subsequent insert
    /// or when the map is dropped.
    #[inline]
    pub fn remove<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.hasher.hash_key(key);
        let fp = (hash >> 57) as u8;

        // Lock-free scan: locate the slot without holding the stripe lock.
        // This keeps the probe scan out of the critical section so the lock
        // is held for an O(1) verify + store rather than a full probe scan.
        let n = self.n_segments.load(Ordering::Acquire);
        let mut hint: Option<(*const Segment<K, V>, usize)> = None;
        for i in 0..n {
            let ptr = self.segments[i].load(Ordering::Relaxed) as *const Segment<K, V>;
            let seg = unsafe { &*ptr };
            if !seg.bloom.maybe_contains(hash) { continue; }
            if let Some(slot_idx) = seg.get_slot_idx(hash, key) {
                hint = Some((ptr, slot_idx));
                break;
            }
        }

        let Some((seg_ptr, hint_slot)) = hint else {
            return false; // key is not present
        };

        // Acquire the stripe lock and atomically tombstone the slot.
        let stripe_idx = hash as usize % NUM_STRIPES;
        let _guard = self.write_stripes[stripe_idx].lock.lock();

        // O(1) verify: same-stripe serialization means the slot cannot be
        // concurrently modified while we hold the lock.  Check fingerprint
        // first (cheap) then key equality.
        let seg = unsafe { &*seg_ptr };
        let meta = seg.meta[hint_slot].load(Ordering::Relaxed);
        if meta == fp {
            // SAFETY: OCCUPIED + stripe lock ⟹ data is stable.
            let (k, _) = unsafe { (*seg.data[hint_slot].get()).assume_init_ref() };
            if k.borrow() == key {
                // Release: pairs with readers' Acquire on the metadata byte.
                seg.meta[hint_slot].store(META_TOMBSTONE, Ordering::Release);
                self.write_stripes[stripe_idx].count.fetch_sub(1, Ordering::Relaxed);
                return true;
            }
        }

        // Rare fallback: the hint slot was reused between our lock-free scan
        // and acquiring the lock (requires a concurrent same-stripe remove + reinsert).
        for i in 0..MAX_SEGMENTS {
            let ptr = self.segments[i].load(Ordering::Acquire);
            if ptr.is_null() {
                break;
            }
            let seg: &Segment<K, V> = unsafe { &*ptr };
            if !seg.bloom.maybe_contains(hash) { continue; }
            if let Ok(slot_idx) = seg.probe_entry(hash, key) {
                seg.meta[slot_idx].store(META_TOMBSTONE, Ordering::Release);
                self.write_stripes[stripe_idx].count.fetch_sub(1, Ordering::Relaxed);
                return true;
            }
        }
        false
    }

    /// Applies `f` to the value associated with `key` under the internal write
    /// lock. Returns `true` if the key was found, `false` otherwise.
    ///
    /// Accepts any borrowed form of the key: `map.modify("str", f)` works for
    /// `HashMap<String, V>` because `String: Borrow<str>`.
    ///
    /// # Concurrency note
    ///
    /// `modify` holds the write lock while `f` runs, serializing it with all
    /// other `insert`, `remove`, and `modify` calls. However, concurrent
    /// lock-free `get()` calls may read the value while `f` mutates it. For
    /// non-atomic `V` this is a data race — only use `modify` when `V` provides
    /// its own synchronization or when no concurrent reads overlap.
    #[inline]
    pub fn modify<Q, F>(&self, key: &Q, f: F) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
        F: FnOnce(&mut V),
    {
        let hash = self.hasher.hash_key(key);
        let _guard = self.write_stripes[hash as usize % NUM_STRIPES].lock.lock();
        for i in 0..MAX_SEGMENTS {
            let ptr = self.segments[i].load(Ordering::Relaxed);
            if ptr.is_null() {
                break;
            }
            let seg: &Segment<K, V> = unsafe { &*ptr };
            if !seg.bloom.maybe_contains(hash) { continue; }
            if let Ok(slot_idx) = seg.probe_entry(hash, key) {
                let v = unsafe {
                    &mut (*seg.data[slot_idx].get()).assume_init_mut().1
                };
                f(v);
                return true;
            }
        }
        false
    }

    // ── Exclusive (&mut self) ─────────────────────────────────────────────────

    /// Removes a key from the map, returning the key-value pair if present.
    ///
    /// Accepts any borrowed form of the key: `map.remove_entry("str")` works
    /// for `HashMap<String, V>` because `String: Borrow<str>`.
    ///
    /// Requires exclusive access (`&mut self`). Use [`remove`](Self::remove)
    /// for the `&self` variant that returns `bool`.
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.hasher.hash_key(key);
        for i in 0..MAX_SEGMENTS {
            let ptr = self.segments[i].load(Ordering::Relaxed);
            if ptr.is_null() {
                break;
            }
            let seg: &Segment<K, V> = unsafe { &*ptr };
            if !seg.bloom.maybe_contains(hash) { continue; }
            if let Ok(slot_idx) = seg.probe_entry(hash, key) {
                let kv = unsafe { (*seg.data[slot_idx].get()).assume_init_read() };
                seg.meta[slot_idx].store(META_EMPTY, Ordering::Relaxed);
                let stripe_idx = hash as usize % NUM_STRIPES;
                self.write_stripes[stripe_idx].count.fetch_sub(1, Ordering::Relaxed);
                return Some(kv);
            }
        }
        None
    }

    /// Clears the map, removing all key-value pairs. Does not free memory.
    pub fn clear(&mut self) {
        for i in 0..MAX_SEGMENTS {
            let ptr = self.segments[i].load(Ordering::Relaxed);
            if ptr.is_null() {
                break;
            }
            let seg: &Segment<K, V> = unsafe { &*ptr };
            for (idx, meta) in seg.meta.iter().enumerate() {
                let b = meta.load(Ordering::Relaxed);
                // Drop OCCUPIED (b & 0x80 == 0) and TOMBSTONE (b == META_TOMBSTONE).
                if b != META_EMPTY && b != META_RESERVED {
                    unsafe { (*seg.data[idx].get()).assume_init_drop() };
                    meta.store(META_EMPTY, Ordering::Relaxed);
                }
            }
        }
        for lane in self.write_stripes.iter() {
            lane.count.store(0, Ordering::Relaxed);
        }
    }

    /// Retains only elements for which `f` returns `true`.
    pub fn retain<F: FnMut(&K, &mut V) -> bool>(&mut self, mut f: F) {
        for i in 0..MAX_SEGMENTS {
            let ptr = self.segments[i].load(Ordering::Relaxed);
            if ptr.is_null() {
                break;
            }
            let seg: &Segment<K, V> = unsafe { &*ptr };
            for (idx, meta) in seg.meta.iter().enumerate() {
                let b = meta.load(Ordering::Relaxed);
                if b == META_TOMBSTONE {
                    // Opportunistically reclaim tombstone slots during retain.
                    unsafe { (*seg.data[idx].get()).assume_init_drop() };
                    meta.store(META_EMPTY, Ordering::Relaxed);
                } else if b & 0x80 == 0 {
                    // OCCUPIED
                    let data_ptr = seg.data[idx].get();
                    let keep = unsafe {
                        let pair: &mut (K, V) = (*data_ptr).assume_init_mut();
                        f(&pair.0, &mut pair.1)
                    };
                    if !keep {
                        // Hash before dropping so the key is still valid.
                        let h = self.hasher.hash_key(unsafe {
                            &(*data_ptr).assume_init_ref().0
                        });
                        unsafe { (*data_ptr).assume_init_drop() };
                        meta.store(META_EMPTY, Ordering::Relaxed);
                        self.write_stripes[h as usize % NUM_STRIPES]
                            .count.fetch_sub(1, Ordering::Relaxed);
                    }
                }
            }
        }
    }

    /// Clears the map, returning all key-value pairs as an iterator.
    ///
    /// The map is left empty when the iterator is dropped, even if not fully
    /// consumed.
    pub fn drain(&mut self) -> Drain<'_, K, V, H> {
        Drain { map: self, seg_idx: 0, slot_idx: 0 }
    }

    /// Gets the entry for `key` for in-place manipulation.
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, H> {
        let hash = self.hasher.hash_key(&key);
        let mut first_empty: Option<(*const Segment<K, V>, usize)> = None;

        for i in 0..MAX_SEGMENTS {
            let ptr = self.segments[i].load(Ordering::Relaxed);
            if ptr.is_null() {
                break;
            }
            let seg: &Segment<K, V> = unsafe { &*ptr };

            if !seg.bloom.maybe_contains(hash) { continue; }

            match seg.probe_entry(hash, &key) {
                Ok(slot_idx) => {
                    return Entry::Occupied(OccupiedEntry {
                        _map: self,
                        seg_ptr: ptr,
                        slot_idx,
                        hash,
                    });
                }
                Err(Some(slot_idx)) => {
                    if first_empty.is_none() {
                        first_empty = Some((ptr, slot_idx));
                    }
                }
                Err(None) => {}
            }
        }

        Entry::Vacant(VacantEntry {
            map: self,
            key,
            hash,
            slot: first_empty,
            _phantom: PhantomData,
        })
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    /// Core insert logic for the exclusive `&mut self` path (used by
    /// [`VacantEntry::insert`] when no pre-found slot is available).
    ///
    /// Returns a raw pointer to the inserted value (valid for the map's lifetime).
    fn insert_locked(&self, key: K, value: V) -> *mut V {
        let hash = self.hasher.hash_key(&key);

        // Key was confirmed absent by the caller (entry() path). Just find a slot.
        let mut first_available: Option<(*const Segment<K, V>, usize)> = None;
        let mut num_segments = 0;

        for i in 0..MAX_SEGMENTS {
            let ptr = self.segments[i].load(Ordering::Relaxed);
            if ptr.is_null() {
                break;
            }
            num_segments = i + 1;

            if first_available.is_none() {
                let seg = unsafe { &*ptr };
                if let Some(slot) = seg.find_available(hash) {
                    first_available = Some((ptr, slot));
                }
            }
        }

        if let Some((seg_ptr, slot_idx)) = first_available {
            let seg = unsafe { &*seg_ptr };
            if seg.meta[slot_idx].load(Ordering::Relaxed) == META_TOMBSTONE {
                unsafe { (*seg.data[slot_idx].get()).assume_init_drop() };
            }
            unsafe { seg.write_slot(slot_idx, hash, key, value) };
            self.write_stripes[hash as usize % NUM_STRIPES]
                .count.fetch_add(1, Ordering::Relaxed);
            return unsafe { &mut (*seg.data[slot_idx].get()).assume_init_mut().1 };
        }

        // All probe sequences exhausted — grow by adding a new segment.
        assert!(num_segments < MAX_SEGMENTS, "anchormap: all segments exhausted (unreachable at normal scales)");

        let last_size = {
            let prev = self.segments[num_segments - 1].load(Ordering::Relaxed);
            unsafe { (*prev).size }
        };
        let new_size = last_size.saturating_mul(2).next_power_of_two();
        let new_seg = Box::new(Segment::new(new_size, true));

        let slot_idx = new_seg
            .find_available(hash)
            .expect("fresh segment must have an empty slot");
        unsafe { new_seg.write_slot(slot_idx, hash, key, value) };

        let val_ptr: *mut V =
            unsafe { &mut (*new_seg.data[slot_idx].get()).assume_init_mut().1 };

        let ptr = Box::into_raw(new_seg);
        self.segments[num_segments].store(ptr, Ordering::Release);
        self.n_segments.fetch_add(1, Ordering::Release);
        self.write_stripes[hash as usize % NUM_STRIPES]
            .count.fetch_add(1, Ordering::Relaxed);
        self.total_capacity.fetch_add(new_size, Ordering::Relaxed);
        val_ptr
    }
}

impl<K, V, H> Drop for HashMap<K, V, H> {
    fn drop(&mut self) {
        for i in 0..MAX_SEGMENTS {
            let ptr = self.segments[i].load(Ordering::Relaxed);
            if ptr.is_null() {
                break;
            }
            // SAFETY: ptr from Box::into_raw; exclusive access via &mut self.
            unsafe { drop(Box::from_raw(ptr)) };
        }
    }
}

// ── Trait impls ───────────────────────────────────────────────────────────────

impl<K, V, H> std::fmt::Debug for HashMap<K, V, H>
where
    K: Eq + Hash + std::fmt::Debug,
    V: std::fmt::Debug,
    H: TableHasher,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K, V> Default for HashMap<K, V, AHashHasher>
where
    K: Eq + Hash,
{
    fn default() -> Self {
        Self::new(16)
    }
}

impl<K, V, H> Clone for HashMap<K, V, H>
where
    K: Eq + Hash + Clone,
    V: Clone,
    H: TableHasher,
{
    fn clone(&self) -> Self {
        let clone = Self::with_hasher(self.len(), self.hasher.clone());
        for (k, v) in self.iter() {
            let _ = clone.insert(k.clone(), v.clone());
        }
        clone
    }
}

impl<K, V, H> PartialEq for HashMap<K, V, H>
where
    K: Eq + Hash,
    V: PartialEq,
    H: TableHasher,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().all(|(k, v)| other.get(k) == Some(v))
    }
}

impl<K, V, H> Eq for HashMap<K, V, H>
where
    K: Eq + Hash,
    V: Eq,
    H: TableHasher,
{}

impl<K, V, H> std::ops::Index<&K> for HashMap<K, V, H>
where
    K: Eq + Hash,
    H: TableHasher,
{
    type Output = V;

    /// # Panics
    ///
    /// Panics if `key` is not present.
    fn index(&self, key: &K) -> &V {
        self.get(key).expect("key not found in HashMap")
    }
}

impl<K, V, H> FromIterator<(K, V)> for HashMap<K, V, H>
where
    K: Eq + Hash,
    H: TableHasher + Default,
{
    /// On duplicate keys the **first** value wins.
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        let cap = upper.unwrap_or(lower).max(16);
        let map = Self::with_hasher(cap, H::default());
        for (k, v) in iter {
            let _ = map.insert(k, v);
        }
        map
    }
}

impl<K, V, H> Extend<(K, V)> for HashMap<K, V, H>
where
    K: Eq + Hash,
    H: TableHasher,
{
    /// On duplicate keys the existing value is kept (first-wins).
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            let _ = self.insert(k, v);
        }
    }
}

// ── IntoIterator impls ────────────────────────────────────────────────────────

impl<'a, K, V, H> IntoIterator for &'a HashMap<K, V, H>
where
    K: Eq + Hash,
    H: TableHasher,
{
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V, H>;

    fn into_iter(self) -> Iter<'a, K, V, H> {
        self.iter()
    }
}

impl<K, V, H> IntoIterator for HashMap<K, V, H> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V, H>;

    fn into_iter(self) -> IntoIter<K, V, H> {
        IntoIter { map: self, seg_idx: 0, slot_idx: 0 }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // ── existing tests ────────────────────────────────────────────────────────

    #[test]
    fn test_insert_and_get() {
        let map = HashMap::<&str, u32>::new(64);
        assert!(map.insert("hello", 42));
        assert_eq!(map.get(&"hello"), Some(&42));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_get_missing_key() {
        let map = HashMap::<&str, u32>::new(64);
        assert_eq!(map.get(&"missing"), None);
    }

    #[test]
    fn test_duplicate_insert_is_noop() {
        let map = HashMap::<&str, u32>::new(64);
        assert!(map.insert("key", 1));
        assert!(!map.insert("key", 2));
        assert_eq!(map.get(&"key"), Some(&1));
    }

    #[test]
    fn test_auto_grow() {
        let map = HashMap::<u32, u32>::new(4);
        let n = 200u32;
        for i in 0..n {
            map.insert(i, i * 7);
        }
        assert_eq!(map.len(), n as usize);
        for i in 0..n {
            assert_eq!(map.get(&i), Some(&(i * 7)), "missing key {i}");
        }
        assert!(map.capacity() > 4);
    }

    #[test]
    fn test_duplicate_across_segments() {
        let map = HashMap::<u32, u32>::new(4);
        for i in 0..100 {
            map.insert(i, i);
        }
        assert!(!map.insert(0, 999));
        assert_eq!(map.get(&0), Some(&0));
    }

    #[test]
    fn test_concurrent_reads() {
        let map = Arc::new(HashMap::<u32, u32>::new(256));
        for i in 0..100 {
            map.insert(i, i * 10);
        }

        std::thread::scope(|s| {
            for _ in 0..8 {
                let map = Arc::clone(&map);
                s.spawn(move || {
                    for i in 0..100u32 {
                        assert_eq!(map.get(&i), Some(&(i * 10)));
                    }
                    for i in 100..200u32 {
                        assert_eq!(map.get(&i), None);
                    }
                });
            }
        });
    }

    #[test]
    fn test_concurrent_inserts() {
        let map = Arc::new(HashMap::<u32, u32>::new(1024));

        std::thread::scope(|s| {
            for t in 0..4u32 {
                let map = Arc::clone(&map);
                s.spawn(move || {
                    let start = t * 50;
                    for i in start..start + 50 {
                        map.insert(i, i);
                    }
                });
            }
        });

        assert_eq!(map.len(), 200);
        for i in 0..200u32 {
            assert_eq!(map.get(&i), Some(&i));
        }
    }

    #[test]
    fn test_concurrent_reads_during_growth() {
        let map = Arc::new(HashMap::<u32, u32>::new(4));

        std::thread::scope(|s| {
            let map_w = Arc::clone(&map);
            s.spawn(move || {
                for i in 0..500u32 {
                    map_w.insert(i, i);
                }
            });

            for _ in 0..4 {
                let map_r = Arc::clone(&map);
                s.spawn(move || {
                    for i in 0..500u32 {
                        if let Some(v) = map_r.get(&i) {
                            assert_eq!(*v, i);
                        }
                    }
                });
            }
        });

        assert_eq!(map.len(), 500);
        for i in 0..500u32 {
            assert_eq!(map.get(&i), Some(&i));
        }
    }

    #[test]
    fn test_with_capacity() {
        let map = HashMap::<u32, u32>::with_capacity(100);
        assert!(map.capacity() >= 100);
        assert!(map.is_empty());
    }

    #[test]
    fn test_iter_basic() {
        let map = HashMap::<u32, u32>::new(64);
        for i in 0..10u32 {
            map.insert(i, i * 2);
        }
        let mut collected: Vec<(u32, u32)> = map.iter().map(|(&k, &v)| (k, v)).collect();
        collected.sort();
        let expected: Vec<(u32, u32)> = (0..10).map(|i| (i, i * 2)).collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn test_iter_across_segments() {
        let map = HashMap::<u32, u32>::new(4);
        for i in 0..100u32 {
            map.insert(i, i);
        }
        let count = map.iter().count();
        assert_eq!(count, 100);
    }

    #[test]
    fn test_iter_for_loop() {
        let map = HashMap::<u32, u32>::new(64);
        for i in 0..5u32 {
            map.insert(i, i * 3);
        }
        let mut sum = 0u32;
        for (_, &v) in &map {
            sum += v;
        }
        assert_eq!(sum, 30);
    }

    #[test]
    fn test_keys_and_values() {
        let map = HashMap::<u32, u32>::new(64);
        for i in 1..=5u32 {
            map.insert(i, i * 10);
        }
        let mut ks: Vec<u32> = map.keys().copied().collect();
        ks.sort();
        assert_eq!(ks, vec![1, 2, 3, 4, 5]);

        let mut vs: Vec<u32> = map.values().copied().collect();
        vs.sort();
        assert_eq!(vs, vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_modify() {
        let map = HashMap::<u32, u32>::new(64);
        map.insert(1, 10);
        assert!(map.modify(&1, |v| *v = 99));
        assert_eq!(map.get(&1), Some(&99));
        assert!(!map.modify(&99, |v| *v = 0));
    }

    #[test]
    fn test_remove_basic() {
        let map = HashMap::<u32, u32>::new(64);
        map.insert(7, 77);
        assert!(map.remove(&7));
        assert_eq!(map.get(&7), None);
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn test_remove_missing() {
        let map = HashMap::<u32, u32>::new(64);
        assert!(!map.remove(&42));
    }

    #[test]
    fn test_remove_entry() {
        let mut map = HashMap::<u32, u32>::new(64);
        map.insert(3, 33);
        assert_eq!(map.remove_entry(&3), Some((3, 33)));
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn test_remove_tombstone_reuse() {
        let map = HashMap::<u32, u32>::new(64);
        map.insert(1, 10);
        assert!(map.remove(&1));
        assert_eq!(map.len(), 0);
        assert!(map.insert(1, 20));
        assert_eq!(map.get(&1), Some(&20));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_remove_tombstone_probe_continues() {
        let map = HashMap::<u32, u32>::new(64);
        for i in 0..50u32 {
            map.insert(i, i * 2);
        }
        for i in 0..25u32 {
            assert!(map.remove(&i));
        }
        assert_eq!(map.len(), 25);
        for i in 25..50u32 {
            assert_eq!(map.get(&i), Some(&(i * 2)), "missing key {i}");
        }
    }

    #[test]
    fn test_concurrent_remove_and_get() {
        let map = Arc::new(HashMap::<u32, u32>::new(256));
        for i in 0..100u32 {
            map.insert(i, i);
        }
        std::thread::scope(|s| {
            let map_r = Arc::clone(&map);
            s.spawn(move || {
                for i in 0..100u32 {
                    let _ = map_r.get(&i);
                }
            });
            let map_w = Arc::clone(&map);
            s.spawn(move || {
                for i in 0..50u32 {
                    map_w.remove(&i);
                }
            });
        });
        assert_eq!(map.len(), 50);
        for i in 50..100u32 {
            assert_eq!(map.get(&i), Some(&i));
        }
    }

    #[test]
    fn test_clear() {
        let mut map = HashMap::<u32, u32>::new(64);
        for i in 0..20u32 {
            map.insert(i, i);
        }
        map.clear();
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        for i in 0..20u32 {
            assert_eq!(map.get(&i), None);
        }
        map.insert(0, 42);
        assert_eq!(map.get(&0), Some(&42));
    }

    #[test]
    fn test_retain_keeps_all() {
        let mut map = HashMap::<u32, u32>::new(64);
        for i in 0..10u32 {
            map.insert(i, i);
        }
        map.retain(|_, _| true);
        assert_eq!(map.len(), 10);
    }

    #[test]
    fn test_retain_filters() {
        let mut map = HashMap::<u32, u32>::new(64);
        for i in 0..10u32 {
            map.insert(i, i);
        }
        map.retain(|&k, _| k % 2 == 0);
        assert_eq!(map.len(), 5);
        for i in 0..10u32 {
            if i % 2 == 0 {
                assert_eq!(map.get(&i), Some(&i));
            } else {
                assert_eq!(map.get(&i), None);
            }
        }
    }

    #[test]
    fn test_entry_vacant_insert() {
        let mut map = HashMap::<u32, u32>::new(64);
        let v = match map.entry(1) {
            Entry::Vacant(e) => e.insert(42),
            Entry::Occupied(_) => panic!("expected vacant"),
        };
        assert_eq!(*v, 42);
        assert_eq!(map.get(&1), Some(&42));
    }

    #[test]
    fn test_entry_occupied_get() {
        let mut map = HashMap::<u32, u32>::new(64);
        map.insert(5, 55);
        match map.entry(5) {
            Entry::Occupied(e) => assert_eq!(*e.get(), 55),
            Entry::Vacant(_) => panic!("expected occupied"),
        }
    }

    #[test]
    fn test_entry_occupied_insert_updates() {
        let mut map = HashMap::<u32, u32>::new(64);
        map.insert(5, 55);
        let old = match map.entry(5) {
            Entry::Occupied(mut e) => e.insert(99),
            Entry::Vacant(_) => panic!("expected occupied"),
        };
        assert_eq!(old, 55);
        assert_eq!(map.get(&5), Some(&99));
    }

    #[test]
    fn test_entry_occupied_remove() {
        let mut map = HashMap::<u32, u32>::new(64);
        map.insert(5, 55);
        let v = match map.entry(5) {
            Entry::Occupied(e) => e.remove(),
            Entry::Vacant(_) => panic!("expected occupied"),
        };
        assert_eq!(v, 55);
        assert_eq!(map.len(), 0);
        assert_eq!(map.get(&5), None);
    }

    #[test]
    fn test_entry_or_insert_pattern() {
        let mut map = HashMap::<u32, u32>::new(64);
        let v = match map.entry(10) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(0),
        };
        *v += 7;
        assert_eq!(map.get(&10), Some(&7));

        let v = match map.entry(10) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(0),
        };
        *v += 3;
        assert_eq!(map.get(&10), Some(&10));
    }

    #[test]
    fn test_into_iter_consuming() {
        let map = HashMap::<u32, u32>::new(64);
        for i in 0..5u32 {
            map.insert(i, i * 2);
        }
        let mut collected: Vec<(u32, u32)> = map.into_iter().collect();
        collected.sort();
        let expected: Vec<(u32, u32)> = (0..5).map(|i| (i, i * 2)).collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn test_with_capacity_and_hasher() {
        use crate::hashing::AHashHasher;
        let map = HashMap::<u32, u32>::with_capacity_and_hasher(64, AHashHasher::default());
        assert!(map.capacity() >= 64);
    }

    #[test]
    fn test_get_key_value() {
        let map = HashMap::<u32, u32>::new(64);
        map.insert(7, 77);
        assert_eq!(map.get_key_value(&7), Some((&7, &77)));
        assert_eq!(map.get_key_value(&99), None);
    }

    #[test]
    fn test_into_keys_values() {
        let map = HashMap::<u32, u32>::new(64);
        for i in 1..=4u32 {
            map.insert(i, i * 10);
        }

        let mut ks: Vec<u32> = HashMap::<u32, u32>::new(64).into_keys().collect();
        ks.sort();
        assert!(ks.is_empty());

        let map2 = HashMap::<u32, u32>::new(64);
        for i in 1..=4u32 {
            map2.insert(i, i * 10);
        }
        let mut vs: Vec<u32> = map2.into_values().collect();
        vs.sort();
        assert_eq!(vs, vec![10, 20, 30, 40]);
    }

    #[test]
    fn test_drain_full() {
        let mut map = HashMap::<u32, u32>::new(64);
        for i in 0..10u32 {
            map.insert(i, i);
        }
        let mut drained: Vec<(u32, u32)> = map.drain().collect();
        drained.sort();
        assert_eq!(drained.len(), 10);
        assert!(map.is_empty());
    }

    #[test]
    fn test_drain_partial_drop() {
        let mut map = HashMap::<u32, String>::new(64);
        for i in 0..10u32 {
            map.insert(i, format!("v{i}"));
        }
        {
            let mut d = map.drain();
            let _ = d.next();
            let _ = d.next();
        }
        assert!(map.is_empty());
    }

    #[test]
    fn test_entry_or_insert() {
        let mut map = HashMap::<u32, u32>::new(64);
        *map.entry(1).or_insert(10) += 5;
        assert_eq!(map.get(&1), Some(&15));
        *map.entry(1).or_insert(99) += 1;
        assert_eq!(map.get(&1), Some(&16));
    }

    #[test]
    fn test_entry_or_default() {
        let mut map = HashMap::<u32, u32>::new(64);
        *map.entry(42).or_default() += 7;
        assert_eq!(map.get(&42), Some(&7));
    }

    #[test]
    fn test_entry_and_modify() {
        let mut map = HashMap::<u32, u32>::new(64);
        map.insert(1, 10);
        map.entry(1).and_modify(|v| *v *= 3).or_insert(0);
        assert_eq!(map.get(&1), Some(&30));
        map.entry(2).and_modify(|v| *v *= 3).or_insert(99);
        assert_eq!(map.get(&2), Some(&99));
    }

    #[test]
    fn test_entry_key() {
        let mut map = HashMap::<u32, u32>::new(64);
        assert_eq!(*map.entry(7).key(), 7);
        map.insert(5, 55);
        assert_eq!(*map.entry(5).key(), 5);
    }

    #[test]
    fn test_debug() {
        let map = HashMap::<u32, u32>::new(64);
        map.insert(1, 10);
        let s = format!("{map:?}");
        assert!(s.contains("10"));
    }

    #[test]
    fn test_default() {
        let map: HashMap<u32, u32> = Default::default();
        assert!(map.is_empty());
    }

    #[test]
    fn test_clone() {
        let map = HashMap::<u32, u32>::new(64);
        for i in 0..20u32 {
            map.insert(i, i * 2);
        }
        let clone = map.clone();
        assert_eq!(clone.len(), map.len());
        for i in 0..20u32 {
            assert_eq!(clone.get(&i), map.get(&i));
        }
    }

    #[test]
    fn test_index() {
        let map = HashMap::<u32, u32>::new(64);
        map.insert(3, 33);
        assert_eq!(map[&3], 33);
    }

    #[test]
    #[should_panic]
    fn test_index_missing_panics() {
        let map = HashMap::<u32, u32>::new(64);
        let _ = map[&99];
    }

    #[test]
    fn test_from_iterator() {
        let pairs: Vec<(u32, u32)> = (0..10).map(|i| (i, i * 2)).collect();
        let map: HashMap<u32, u32> = pairs.into_iter().collect();
        assert_eq!(map.len(), 10);
        assert_eq!(map.get(&5), Some(&10));
    }

    #[test]
    fn test_from_iterator_duplicate_first_wins() {
        let pairs = vec![(1u32, 10u32), (1, 20), (1, 30)];
        let map: HashMap<u32, u32> = pairs.into_iter().collect();
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&10));
    }

    #[test]
    fn test_extend() {
        let mut map = HashMap::<u32, u32>::new(64);
        map.insert(0, 0);
        map.extend((1..5).map(|i| (i, i * 10)));
        assert_eq!(map.len(), 5);
        assert_eq!(map.get(&3), Some(&30));
    }

    #[test]
    fn test_partial_eq_equal() {
        let a = HashMap::<u32, u32>::new(64);
        let b = HashMap::<u32, u32>::new(64);
        for i in 0..5u32 {
            a.insert(i, i * 2);
        }
        for i in 0..5u32 {
            b.insert(i, i * 2);
        }
        assert_eq!(a, b);
    }

    #[test]
    fn test_partial_eq_different_values() {
        let a = HashMap::<u32, u32>::new(64);
        let b = HashMap::<u32, u32>::new(64);
        a.insert(1, 10);
        b.insert(1, 99);
        assert_ne!(a, b);
    }

    #[test]
    fn test_partial_eq_different_keys() {
        let a = HashMap::<u32, u32>::new(64);
        let b = HashMap::<u32, u32>::new(64);
        a.insert(1, 10);
        b.insert(2, 10);
        assert_ne!(a, b);
    }

    #[test]
    fn test_partial_eq_different_len() {
        let a = HashMap::<u32, u32>::new(64);
        let b = HashMap::<u32, u32>::new(64);
        a.insert(1, 10);
        assert_ne!(a, b);
    }

    #[test]
    fn test_entry_types_named_entry() {
        let mut map = HashMap::<u32, u32>::new(64);
        map.insert(1, 10);
        match map.entry(1) {
            Entry::Occupied(e) => assert_eq!(*e.get(), 10),
            Entry::Vacant(_) => panic!("expected occupied"),
        }
        match map.entry(99) {
            Entry::Occupied(_) => panic!("expected vacant"),
            Entry::Vacant(e) => {
                e.insert(99);
            }
        }
    }

    #[test]
    fn test_into_iter_partial_drop() {
        let map = HashMap::<u32, String>::new(64);
        for i in 0..20u32 {
            map.insert(i, format!("val-{i}"));
        }
        let mut it = map.into_iter();
        let _ = it.next();
        let _ = it.next();
        drop(it);
    }

    // ── Borrow<Q> tests ───────────────────────────────────────────────────────

    #[test]
    fn test_borrow_get_string_key() {
        let map = HashMap::<String, u32>::new(64);
        map.insert("hello".to_string(), 42);
        // get, contains_key, get_key_value accept &str for HashMap<String, _>
        assert_eq!(map.get("hello"), Some(&42));
        assert_eq!(map.get("missing"), None);
        assert!(map.contains_key("hello"));
        assert!(!map.contains_key("missing"));
        let (k, v) = map.get_key_value("hello").unwrap();
        assert_eq!(k, "hello");
        assert_eq!(*v, 42);
    }

    #[test]
    fn test_borrow_remove_string_key() {
        let map = HashMap::<String, u32>::new(64);
        map.insert("alpha".to_string(), 1);
        map.insert("beta".to_string(), 2);
        assert!(map.remove("alpha"));
        assert_eq!(map.get("alpha"), None);
        assert_eq!(map.get("beta"), Some(&2));
    }

    #[test]
    fn test_borrow_remove_entry_string_key() {
        let mut map = HashMap::<String, u32>::new(64);
        map.insert("key".to_string(), 99);
        let pair = map.remove_entry("key");
        assert_eq!(pair, Some(("key".to_string(), 99)));
        assert!(map.is_empty());
    }

    #[test]
    fn test_borrow_modify_string_key() {
        let map = HashMap::<String, u32>::new(64);
        map.insert("x".to_string(), 10);
        assert!(map.modify("x", |v| *v += 5));
        assert_eq!(map.get("x"), Some(&15));
        assert!(!map.modify("absent", |v| *v += 1));
    }

    // ── get_or_insert / get_or_insert_with tests ──────────────────────────────

    #[test]
    fn test_get_or_insert_absent() {
        let map = HashMap::<u32, u32>::new(64);
        let v = map.get_or_insert(1, 42);
        assert_eq!(*v, 42);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_get_or_insert_present() {
        let map = HashMap::<u32, u32>::new(64);
        map.insert(1, 10);
        let v = map.get_or_insert(1, 99);
        // key already present — original value returned, map unchanged
        assert_eq!(*v, 10);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_get_or_insert_with_absent() {
        let map = HashMap::<u32, u32>::new(64);
        let mut called = false;
        let v = map.get_or_insert_with(7, || { called = true; 77 });
        assert!(called);
        assert_eq!(*v, 77);
    }

    #[test]
    fn test_get_or_insert_with_present_does_not_call_closure() {
        let map = HashMap::<u32, u32>::new(64);
        map.insert(7, 77);
        let mut called = false;
        let v = map.get_or_insert_with(7, || { called = true; 0 });
        assert!(!called, "closure must not be called when key is present");
        assert_eq!(*v, 77);
    }

    #[test]
    fn test_get_or_insert_reference_lifetime() {
        // The reference returned by get_or_insert is valid for the map's lifetime.
        let map = HashMap::<u32, String>::new(64);
        let r: &String = map.get_or_insert(1, "hello".to_string());
        // Force a concurrent insert of a different key to exercise growth.
        for i in 2..50u32 {
            map.insert(i, format!("v{i}"));
        }
        // r is still valid — values never move after insertion.
        assert_eq!(r, "hello");
    }

    #[test]
    fn test_get_or_insert_concurrent() {
        let map = Arc::new(HashMap::<u32, u32>::new(64));
        std::thread::scope(|s| {
            for _ in 0..8 {
                let map = Arc::clone(&map);
                s.spawn(move || {
                    let v = map.get_or_insert(42, 1);
                    assert_eq!(*v, 1); // only one value can win
                });
            }
        });
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&42), Some(&1));
    }
}
