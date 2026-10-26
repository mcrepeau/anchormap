//! Randomized stress and soak tests — the third verification leg after the
//! exhaustive Loom models and the Miri runs.
//!
//! - `distinct_key_ownership`: every thread exclusively owns a disjoint key
//!   range, so each operation's result is exactly predictable from a
//!   thread-local model. Exercises concurrent growth and heavy cross-stripe
//!   write traffic with a per-op oracle.
//! - `shared_key_linearizability`: random get/insert/remove operations on a
//!   small shared key space from many threads. Each op's result is recorded
//!   together with a real-time interval (a global logical clock is read
//!   before and after the call); afterwards each key's history is checked
//!   for linearizability against a sequential model with a backtracking
//!   search (Wing–Gong style).
//! - `held_reference_stability`: readers acquire `&V` and hold it across
//!   unrelated concurrent churn, then verify the referenced value never
//!   changed — the anchored-slot invariant under load.
//!
//! The heavy variant `long_soak` is `#[ignore]`d; run it with:
//!
//! ```text
//! cargo test --test stress -- --ignored
//! ```

#![cfg(not(any(miri, loom)))]

use std::collections::{HashMap as StdMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use anchormap::HashMap;

// ── tiny deterministic PRNG ────────────────────────────────────────────────
//
// Deterministic op sequences per (seed, thread) keep failures reproducible:
// the seed fully determines which operations each thread attempts; only the
// interleaving varies between runs.

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

// ── distinct key ownership ────────────────────────────────────────────────

/// Exclusive key ownership: every thread owns `[t * keys_per_thread,
/// (t+1) * keys_per_thread)`, so all its results follow deterministically
/// from the thread-local `present` set.
fn run_ownership(threads: u64, keys_per_thread: u64, churn: u64) {
    let map = HashMap::<u64, u64>::new(64); // small start: forces growth
    let lens: Vec<usize> = thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let map = &map;
                s.spawn(move || {
                    let mut rng = Rng::new(0xC0FF_EE00 ^ t);
                    let mut present: HashSet<u64> = HashSet::new();
                    let key = |i: u64| t * keys_per_thread + i;

                    // Phase 1: populate this thread's range.
                    for i in 0..keys_per_thread {
                        assert!(map.insert(key(i), key(i)), "fresh key must insert");
                        present.insert(key(i));
                    }

                    // Phase 2: random churn with per-op oracle.
                    for _ in 0..churn {
                        let k = key(rng.below(keys_per_thread));
                        match rng.below(4) {
                            0 | 1 => {
                                let expected = if present.contains(&k) { Some(k) } else { None };
                                assert_eq!(map.get(&k).copied(), expected, "get({k})");
                            }
                            2 => {
                                let expected = present.contains(&k);
                                assert_eq!(map.remove(&k), expected, "remove({k})");
                                present.remove(&k);
                            }
                            _ => {
                                let expected = !present.contains(&k);
                                assert_eq!(map.insert(k, k), expected, "insert({k})");
                                if expected {
                                    present.insert(k);
                                }
                            }
                        }
                    }

                    // Phase 3: everything this thread believes present is
                    // still present with the right value.
                    for &k in &present {
                        assert_eq!(map.get(&k).copied(), Some(k));
                    }
                    present.len()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let total: usize = lens.iter().sum();
    assert_eq!(map.len(), total, "len() must match summed thread models");
}

#[test]
fn distinct_key_ownership() {
    run_ownership(8, 2048, 2048);
}

// ── per-key linearizability ───────────────────────────────────────────────

/// One recorded operation. `t0`/`t1` are logical-clock ticks taken
/// immediately before and after the call, giving a real-time interval.
#[derive(Clone, Copy)]
struct Op {
    kind: u8, // 0 = get, 1 = insert, 2 = remove
    key: u64,
    arg: u64, // insert value: (thread << 32) | op index
    bool_res: bool,
    val_res: Option<u64>, // get result
    t0: usize,
    t1: usize,
}

fn run_linearizability(seed: u64, threads: u64, ops: u64, keys: u64) {
    let map = HashMap::<u64, u64>::new(64);
    let clock = AtomicUsize::new(0);
    let histories: Vec<Vec<Op>> = thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let map = &map;
                let clock = &clock;
                s.spawn(move || {
                    let mut rng = Rng::new(seed ^ t.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    let mut hist: Vec<Op> = Vec::with_capacity(ops as usize);
                    for i in 0..ops {
                        let key = rng.below(keys);
                        let op = match rng.below(10) {
                            0..=3 => {
                                let t0 = clock.fetch_add(1, Ordering::Relaxed);
                                let res = map.get(&key).copied();
                                let t1 = clock.fetch_add(1, Ordering::Relaxed);
                                Op { kind: 0, key, arg: 0, bool_res: false, val_res: res, t0, t1 }
                            }
                            4..=7 => {
                                let val = (t << 32) | i;
                                let t0 = clock.fetch_add(1, Ordering::Relaxed);
                                let res = map.insert(key, val);
                                let t1 = clock.fetch_add(1, Ordering::Relaxed);
                                Op { kind: 1, key, arg: val, bool_res: res, val_res: None, t0, t1 }
                            }
                            _ => {
                                let t0 = clock.fetch_add(1, Ordering::Relaxed);
                                let res = map.remove(&key);
                                let t1 = clock.fetch_add(1, Ordering::Relaxed);
                                Op { kind: 2, key, arg: 0, bool_res: res, val_res: None, t0, t1 }
                            }
                        };
                        hist.push(op);
                    }
                    hist
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Check each key's history independently.
    let mut by_key: StdMap<u64, Vec<Op>> = StdMap::new();
    for op in histories.into_iter().flatten() {
        by_key.entry(op.key).or_default().push(op);
    }
    for (key, key_ops) in &mut by_key {
        key_ops.sort_unstable_by_key(|o| o.t0);
        let mut placed = vec![false; key_ops.len()];
        let mut nodes = 0u64;
        assert!(
            linearizable(key_ops, &mut placed, None, &mut nodes),
            "seed {seed}: history for key {key} is not linearizable"
        );
    }
}

/// Backtracking linearizability search for one key (Wing–Gong style).
///
/// At each step a recorded op may be appended to the linearization order if:
/// 1. no unplaced op that finished before it started must go first
///    (real-time constraint), and
/// 2. the op's recorded result matches the sequential model's state, and
///    applying it produces the next state.
fn linearizable(ops: &[Op], placed: &mut [bool], state: Option<u64>, nodes: &mut u64) -> bool {
    *nodes += 1;
    assert!(
        *nodes < 10_000_000,
        "linearizability checker node budget exceeded; reduce per-key op count"
    );
    if placed.iter().all(|&p| p) {
        return true;
    }
    for i in 0..ops.len() {
        if placed[i] {
            continue;
        }
        // Real-time constraint.
        if ops.iter().zip(placed.iter()).any(|(o, &p)| !p && o.t1 < ops[i].t0) {
            continue;
        }
        let next = match ops[i].kind {
            // get: recorded result must equal the model state.
            0 => {
                if ops[i].val_res == state { state } else { continue; }
            }
            // insert: true requires absence and installs the argument;
            // false requires presence and changes nothing.
            1 => match state {
                None if ops[i].bool_res => Some(ops[i].arg),
                Some(_) if !ops[i].bool_res => state,
                _ => continue,
            },
            // remove: true requires presence and clears; false requires
            // absence.
            _ => match state {
                Some(_) if ops[i].bool_res => None,
                None if !ops[i].bool_res => None,
                _ => continue,
            },
        };
        placed[i] = true;
        if linearizable(ops, placed, next, nodes) {
            return true;
        }
        placed[i] = false;
    }
    false
}

#[test]
fn shared_key_linearizability() {
    for seed in [1u64, 2, 3, 4] {
        run_linearizability(seed, 8, 256, 64);
    }
}

// ── held &V stability across churn ────────────────────────────────────────

fn run_held_refs(writers: u64, writer_ops: u64, readers: u64, holds: u64, hold_ops: u64, keys: u64) {
    let map = HashMap::<u64, u64>::new(64);
    for k in 0..keys {
        map.insert(k, k);
    }

    thread::scope(|s| {
        // Writers churn the shared key space.
        for w in 0..writers {
            let map = &map;
            s.spawn(move || {
                let mut rng = Rng::new(0xBEEF_0000 ^ w);
                for _ in 0..writer_ops {
                    let k = rng.below(keys);
                    if rng.below(2) == 0 {
                        map.remove(&k);
                    } else {
                        map.insert(k, k);
                    }
                }
            });
        }
        // Readers acquire &V, hold it across unrelated churn (which may
        // tombstone the very slot the reference points at), and verify the
        // referenced value is untouched.
        for r in 0..readers {
            let map = &map;
            s.spawn(move || {
                let mut rng = Rng::new(0xFACE_0000 ^ r);
                for _ in 0..holds {
                    let k = rng.below(keys);
                    if let Some(v) = map.get(&k) {
                        let expected = *v;
                        for _ in 0..hold_ops {
                            let k2 = rng.below(keys);
                            let _ = map.get(&k2);
                            if rng.below(4) == 0 {
                                // May tombstone v's slot — must not affect *v.
                                map.remove(&k2);
                            }
                        }
                        assert_eq!(*v, expected, "held &V changed across churn");
                    }
                }
            });
        }
    });

    // Final consistency: every present key still maps to itself.
    for k in 0..keys {
        assert!(
            map.get(&k).is_none_or(|&x| x == k),
            "corrupt value for key {k}"
        );
    }
}

#[test]
fn held_reference_stability() {
    run_held_refs(4, 5000, 4, 64, 256, 512);
}

// ── long soak ──────────────────────────────────────────────────────────────

#[test]
#[ignore = "long-running soak; run with: cargo test --test stress -- --ignored"]
fn long_soak() {
    run_ownership(16, 65536, 32768);
    run_held_refs(8, 500_000, 8, 256, 512, 4096);
    for seed in 1..=24u64 {
        run_linearizability(seed, 16, 4096, 256);
    }
}
