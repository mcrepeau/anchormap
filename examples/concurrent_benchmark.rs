//! Concurrent throughput benchmark — fair, cache-realistic parameters
//!
//! Compares anchormap::HashMap (lock-free reads) vs RwLock<StdHashMap> vs
//! DashMap vs Papaya across four workload profiles and multiple thread counts.
//!
//! Design decisions:
//! - 10 M preloaded keys (≈160 MB of data) → exceeds L3 on most machines,
//!   so every miss costs a real DRAM round-trip.
//! - xorshift64 PRNG per thread for random key access (not sequential).
//! - Four operations: get, insert (fresh key), remove, modify (update in place).
//! - Workloads aligned with conc-map-bench for comparability.
//! - Reads and writes reported separately (not combined totals).
//!
//! Run with:  cargo run --release --example concurrent_benchmark

use std::collections::HashMap as StdHashMap;
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use papaya::HashMap as PapayaMap;
use anchormap::HashMap;

// ── Parameters ───────────────────────────────────────────────────────────────

/// Number of keys preloaded before each benchmark run.
/// At 16 bytes per (u64, u64) entry this is ≈160 MB.
const INITIAL_KEYS: u64 = 10_000_000;

/// Initial capacity: ~70 % load factor.
const CAPACITY: usize = (INITIAL_KEYS as usize) + 4_000_000;

/// Wall-clock budget per (workload, implementation, thread-count) cell.
const RUN_DURATION: Duration = Duration::from_millis(1000);

// ── xorshift64 PRNG ──────────────────────────────────────────────────────────

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

// ── Map trait ────────────────────────────────────────────────────────────────

trait ConcurrentMap: Send + Sync + 'static {
    fn get_hit(&self, key: u64) -> bool;
    fn insert_new(&self, key: u64);
    fn remove_key(&self, key: u64);
    /// Update the value of an existing key in place.  No-op if the key is absent.
    fn modify_key(&self, key: u64);
}

// ── anchormap ───────────────────────────────────────────────────────────────────

// AnchorMap stores AtomicU64 values for correct concurrent in-place mutation:
// get() returns &AtomicU64, then fetch_add is lock-free and data-race-free.
struct AnchorMap(HashMap<u64, AtomicU64>);

impl ConcurrentMap for AnchorMap {
    #[inline]
    fn get_hit(&self, key: u64) -> bool {
        self.0.get(&key).is_some()
    }
    #[inline]
    fn insert_new(&self, key: u64) {
        let _ = self.0.insert(key, AtomicU64::new(key));
    }
    #[inline]
    fn remove_key(&self, key: u64) {
        self.0.remove(&key);
    }
    #[inline]
    fn modify_key(&self, key: u64) {
        if let Some(v) = self.0.get(&key) {
            v.fetch_add(1, Ordering::Relaxed);
        }
    }
}

// ── RwLock<StdHashMap> ───────────────────────────────────────────────────────

struct RwHashMap(RwLock<StdHashMap<u64, u64>>);

impl ConcurrentMap for RwHashMap {
    #[inline]
    fn get_hit(&self, key: u64) -> bool {
        self.0.read().unwrap().get(&key).is_some()
    }
    #[inline]
    fn insert_new(&self, key: u64) {
        self.0.write().unwrap().insert(key, key);
    }
    #[inline]
    fn remove_key(&self, key: u64) {
        self.0.write().unwrap().remove(&key);
    }
    #[inline]
    fn modify_key(&self, key: u64) {
        if let Some(v) = self.0.write().unwrap().get_mut(&key) {
            *v = v.wrapping_add(1);
        }
    }
}

// ── DashMap ───────────────────────────────────────────────────────────────────

struct Dash(DashMap<u64, u64>);

impl ConcurrentMap for Dash {
    #[inline]
    fn get_hit(&self, key: u64) -> bool {
        self.0.get(&key).is_some()
    }
    #[inline]
    fn insert_new(&self, key: u64) {
        self.0.insert(key, key);
    }
    #[inline]
    fn remove_key(&self, key: u64) {
        self.0.remove(&key);
    }
    #[inline]
    fn modify_key(&self, key: u64) {
        if let Some(mut v) = self.0.get_mut(&key) {
            *v = v.wrapping_add(1);
        }
    }
}

// ── Papaya ────────────────────────────────────────────────────────────────────

struct Papaya(PapayaMap<u64, u64>);

impl ConcurrentMap for Papaya {
    #[inline]
    fn get_hit(&self, key: u64) -> bool {
        self.0.pin().get(&key).is_some()
    }
    #[inline]
    fn insert_new(&self, key: u64) {
        self.0.pin().insert(key, key);
    }
    #[inline]
    fn remove_key(&self, key: u64) {
        self.0.pin().remove(&key);
    }
    #[inline]
    fn modify_key(&self, key: u64) {
        self.0.pin().update(key, |v| v.wrapping_add(1));
    }
}

// ── Workload ─────────────────────────────────────────────────────────────────

struct Workload {
    label: &'static str,
    /// Percentage of operations that are gets (0–100). Targets preloaded keys.
    get_pct: u64,
    /// Percentage of operations that are inserts (0–100). Always fresh keys.
    insert_pct: u64,
    /// Percentage of operations that are removes (0–100). Targets preloaded keys.
    remove_pct: u64,
    // modify_pct = 100 - get_pct - insert_pct - remove_pct. Targets preloaded keys.
}

/// Workloads aligned with conc-map-bench for direct comparability.
const WORKLOADS: &[Workload] = &[
    // Nearly all reads — mirrors conc-map-bench ReadHeavy (98/1/1/0).
    Workload { label: "read-heavy",  get_pct: 98, insert_pct: 1,  remove_pct: 1  },
    // Balanced writes — mirrors conc-map-bench WriteHeavy (20/10/10/60).
    Workload { label: "write-heavy", get_pct: 20, insert_pct: 10, remove_pct: 10 },
    // Insert/remove dominant — mirrors conc-map-bench Exchange (10/40/40/10).
    Workload { label: "exchange",    get_pct: 10, insert_pct: 40, remove_pct: 40 },
    // Mostly inserts — mirrors conc-map-bench RapidGrow (5/80/5/10).
    Workload { label: "rapid-grow",  get_pct: 5,  insert_pct: 80, remove_pct: 5  },
];

// ── Runner ────────────────────────────────────────────────────────────────────

struct RunResult {
    reads: u64,
    writes: u64,
    elapsed: Duration,
}

/// Spawn `n_threads` threads that each independently choose an operation
/// according to `workload`, running for `RUN_DURATION`.
///
/// Operation dispatch (r = xorshift64 % 100):
///   r < get_pct                                → get (preloaded key range)
///   r < get_pct + insert_pct                   → insert (globally unique fresh key)
///   r < get_pct + insert_pct + remove_pct      → remove (preloaded key range)
///   otherwise                                  → modify (preloaded key range)
fn run<M: ConcurrentMap>(map: Arc<M>, n_threads: usize, wl: &Workload) -> RunResult {
    let stop = Arc::new(AtomicBool::new(false));
    let total_reads  = Arc::new(AtomicU64::new(0));
    let total_writes = Arc::new(AtomicU64::new(0));
    // Fresh-key counter: inserts always use a globally unique key.
    let write_counter = Arc::new(AtomicU64::new(INITIAL_KEYS));

    let get_pct    = wl.get_pct;
    let insert_pct = wl.insert_pct;
    let remove_pct = wl.remove_pct;

    let start = Instant::now();

    std::thread::scope(|s| {
        for t in 0..n_threads {
            let map           = Arc::clone(&map);
            let stop          = Arc::clone(&stop);
            let total_reads   = Arc::clone(&total_reads);
            let total_writes  = Arc::clone(&total_writes);
            let write_counter = Arc::clone(&write_counter);

            s.spawn(move || {
                // Each thread gets a unique PRNG seed.
                let mut rng: u64 = (t as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                if rng == 0 { rng = 1; }

                let mut local_reads:  u64 = 0;
                let mut local_writes: u64 = 0;

                while !stop.load(Ordering::Relaxed) {
                    // Batch 64 ops before checking the stop flag.
                    for _ in 0..64 {
                        let r = xorshift64(&mut rng) % 100;
                        if r < get_pct {
                            let key = xorshift64(&mut rng) % INITIAL_KEYS;
                            black_box(map.get_hit(key));
                            local_reads += 1;
                        } else if r < get_pct + insert_pct {
                            let key = write_counter.fetch_add(1, Ordering::Relaxed);
                            map.insert_new(key);
                            local_writes += 1;
                        } else if r < get_pct + insert_pct + remove_pct {
                            let key = xorshift64(&mut rng) % INITIAL_KEYS;
                            map.remove_key(key);
                            local_writes += 1;
                        } else {
                            let key = xorshift64(&mut rng) % INITIAL_KEYS;
                            map.modify_key(key);
                            local_writes += 1;
                        }
                    }
                }

                total_reads.fetch_add(local_reads,   Ordering::Relaxed);
                total_writes.fetch_add(local_writes, Ordering::Relaxed);
            });
        }

        std::thread::sleep(RUN_DURATION);
        stop.store(true, Ordering::Relaxed);
    });

    RunResult {
        reads:   total_reads.load(Ordering::Relaxed),
        writes:  total_writes.load(Ordering::Relaxed),
        elapsed: start.elapsed(),
    }
}

// ── Preloaders ────────────────────────────────────────────────────────────────

fn preload_anchormap() -> AnchorMap {
    let map = HashMap::new(CAPACITY);
    for i in 0..INITIAL_KEYS {
        map.insert(i, AtomicU64::new(i));
    }
    AnchorMap(map)
}

fn preload_rwlock() -> RwHashMap {
    let mut inner = StdHashMap::with_capacity(CAPACITY);
    for i in 0..INITIAL_KEYS {
        inner.insert(i, i);
    }
    RwHashMap(RwLock::new(inner))
}

fn preload_dash() -> Dash {
    let map = DashMap::with_capacity(CAPACITY);
    for i in 0..INITIAL_KEYS {
        map.insert(i, i);
    }
    Dash(map)
}

fn preload_papaya() -> Papaya {
    let map = PapayaMap::with_capacity(CAPACITY);
    {
        let guard = map.pin();
        for i in 0..INITIAL_KEYS {
            guard.insert(i, i);
        }
    }
    Papaya(map)
}

// ── Formatting helpers ────────────────────────────────────────────────────────

fn mops(ops: u64, elapsed: Duration) -> f64 {
    ops as f64 / elapsed.as_secs_f64() / 1_000_000.0
}

fn print_header() {
    println!(
        "\n{:<12} {:>3} | {:>8} {:>8} | {:>8} {:>8} | {:>8} {:>8} | {:>8} {:>8}",
        "Workload", "T",
        "Anch-R", "Anch-W",
        "Papaya-R", "Papaya-W",
        "Dash-R", "Dash-W",
        "RwLk-R", "RwLk-W",
    );
    println!("{}", "-".repeat(100));
}

fn run_row(wl: &Workload, n_threads: usize) {
    let r_anch  = run(Arc::new(preload_anchormap()), n_threads, wl);
    let r_pap  = run(Arc::new(preload_papaya()),  n_threads, wl);
    let r_dash = run(Arc::new(preload_dash()),     n_threads, wl);
    let r_rw   = run(Arc::new(preload_rwlock()),   n_threads, wl);

    println!(
        "{:<12} {:>3} | {:>8.1} {:>8.1} | {:>8.1} {:>8.1} | {:>8.1} {:>8.1} | {:>8.1} {:>8.1}",
        wl.label,
        n_threads,
        mops(r_anch.reads,  r_anch.elapsed),
        mops(r_anch.writes, r_anch.elapsed),
        mops(r_pap.reads,  r_pap.elapsed),
        mops(r_pap.writes, r_pap.elapsed),
        mops(r_dash.reads, r_dash.elapsed),
        mops(r_dash.writes,r_dash.elapsed),
        mops(r_rw.reads,   r_rw.elapsed),
        mops(r_rw.writes,  r_rw.elapsed),
    );
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let ncpu = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    println!("═══ Concurrent Map Throughput Benchmark ═══");
    println!("  Initial keys : {INITIAL_KEYS}");
    println!("  Capacity     : {CAPACITY}");
    println!("  Run duration : {}ms per cell", RUN_DURATION.as_millis());
    println!("  Logical CPUs : {ncpu}");
    println!("  Columns      : R = read Mop/s, W = write Mop/s (separate)");
    println!("  Operations   : get (preloaded range), insert (fresh key),");
    println!("                 remove (preloaded range), modify/update (preloaded range)");

    let thread_counts: Vec<usize> = [1, 2, 4, 8]
        .iter()
        .copied()
        .filter(|&t| t <= ncpu)
        .collect();

    // ── Section 1: per-workload thread scaling ────────────────────────────────
    for wl in WORKLOADS {
        println!("\n── {} ──────────────────────────────────────────────────────", wl.label);
        println!(
            "  get={}%  insert={}%  remove={}%  modify={}%",
            wl.get_pct, wl.insert_pct, wl.remove_pct,
            100 - wl.get_pct - wl.insert_pct - wl.remove_pct
        );
        print_header();
        for &t in &thread_counts {
            run_row(wl, t);
        }
    }

    println!("\n═══ Done ═══\n");
}
