//! Benchmarks for `anchormap::HashMap` — concurrent, lock-free reads.
//!
//! Run:
//!   cargo bench
//!   cargo bench -- map_insert   # filter by name

use std::hint::black_box;
use std::sync::Arc;
use std::thread;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use anchormap::HashMap;

// ── shared helpers ────────────────────────────────────────────────────────────

fn filled_map(n: usize) -> HashMap<u64, u64> {
    let m = HashMap::new(n);
    for i in 0..n as u64 {
        m.insert(i, i * 2);
    }
    m
}

// ─────────────────────────────────────────────────────────────────────────────
// Insert
// ─────────────────────────────────────────────────────────────────────────────

fn map_insert(c: &mut Criterion) {
    let mut g = c.benchmark_group("anchormap/insert");
    for n in [1_000u64, 10_000] {
        g.throughput(Throughput::Elements(n));
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let m = HashMap::<u64, u64>::new(n as usize);
                for i in 0..n {
                    m.insert(black_box(i), i);
                }
            });
        });
    }
    g.finish();
}

fn map_insert_growing(c: &mut Criterion) {
    // Start with capacity 1 so the map must grow multiple times.
    c.bench_function("anchormap/insert_growing_1k", |b| {
        b.iter(|| {
            let m = HashMap::<u64, u64>::new(1);
            for i in 0..1_000u64 {
                m.insert(black_box(i), i);
            }
        });
    });
}

/// Insert a key that is already present — exercises the combined
/// dup-check + early-return path without writing any slot.
fn map_insert_dup(c: &mut Criterion) {
    let n = 10_000usize;
    let m = filled_map(n);
    let mut g = c.benchmark_group("anchormap/insert_dup");
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("10k", |b| {
        b.iter(|| {
            let mut rejected = 0usize;
            for i in 0..n as u64 {
                if !m.insert(black_box(i), i) {
                    rejected += 1;
                }
            }
            black_box(rejected);
        });
    });
    g.finish();
}

fn map_concurrent_insert(c: &mut Criterion) {
    let mut g = c.benchmark_group("anchormap/concurrent_insert");
    let n = 10_000usize;
    for &threads in &[2usize, 4, 8] {
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("threads", threads), &threads, |b, &threads| {
            b.iter(|| {
                let m = Arc::new(HashMap::<u64, u64>::new(n * threads));
                let per = n / threads;
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let m = Arc::clone(&m);
                        let start = (t * per) as u64;
                        thread::spawn(move || {
                            for i in start..start + per as u64 {
                                m.insert(black_box(i), i);
                            }
                        })
                    })
                    .collect();
                for h in handles {
                    h.join().unwrap();
                }
            });
        });
    }
    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Get
// ─────────────────────────────────────────────────────────────────────────────

fn map_get_hit(c: &mut Criterion) {
    let n = 10_000usize;
    let m = filled_map(n);
    let mut g = c.benchmark_group("anchormap/get_hit");
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("10k_single_thread", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for i in 0..n as u64 {
                sum = sum.wrapping_add(*m.get(black_box(&i)).unwrap());
            }
            black_box(sum);
        });
    });
    g.finish();
}

fn map_get_miss(c: &mut Criterion) {
    let n = 10_000usize;
    let m = filled_map(n);
    let mut g = c.benchmark_group("anchormap/get_miss");
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("10k_single_thread", |b| {
        b.iter(|| {
            let mut found = 0usize;
            for i in n as u64..(2 * n) as u64 {
                if m.get(black_box(&i)).is_some() {
                    found += 1;
                }
            }
            black_box(found);
        });
    });
    g.finish();
}

fn map_get_concurrent(c: &mut Criterion) {
    let n = 10_000usize;
    let m = Arc::new(filled_map(n));
    let mut g = c.benchmark_group("anchormap/get_concurrent");
    g.throughput(Throughput::Elements(n as u64));
    for &threads in &[2usize, 4, 8] {
        g.bench_with_input(BenchmarkId::new("threads", threads), &threads, |b, &threads| {
            b.iter(|| {
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let m = Arc::clone(&m);
                        thread::spawn(move || {
                            let mut sum = 0u64;
                            for i in 0..n as u64 {
                                sum = sum.wrapping_add(*m.get(black_box(&i)).unwrap());
                            }
                            black_box(sum);
                        })
                    })
                    .collect();
                for h in handles {
                    h.join().unwrap();
                }
            });
        });
    }
    g.finish();
}

fn map_contains_key(c: &mut Criterion) {
    let n = 10_000usize;
    let m = filled_map(n);
    c.bench_function("anchormap/contains_key_hit_10k", |b| {
        b.iter(|| {
            let mut found = 0usize;
            for i in 0..n as u64 {
                if m.contains_key(black_box(&i)) {
                    found += 1;
                }
            }
            black_box(found);
        });
    });
}

fn map_get_key_value(c: &mut Criterion) {
    let n = 10_000usize;
    let m = filled_map(n);
    let mut g = c.benchmark_group("anchormap/get_key_value");
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("hit_10k", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for i in 0..n as u64 {
                let (k, v) = m.get_key_value(black_box(&i)).unwrap();
                sum = sum.wrapping_add(*k).wrapping_add(*v);
            }
            black_box(sum);
        });
    });
    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Mutation
// ─────────────────────────────────────────────────────────────────────────────

fn map_remove(c: &mut Criterion) {
    let n = 1_000usize;
    let mut g = c.benchmark_group("anchormap/remove");
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("1k", |b| {
        b.iter_batched(
            || filled_map(n),
            |m| {
                for i in 0..n as u64 {
                    black_box(m.remove(black_box(&i)));
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
    g.finish();
}

/// Remove every key then reinsert it — exercises the tombstone-reuse slot
/// claim path, which is the main write path for stable-size maps.
fn map_remove_reinsert(c: &mut Criterion) {
    let n = 1_000usize;
    let mut g = c.benchmark_group("anchormap/remove_reinsert");
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("1k", |b| {
        b.iter_batched(
            || filled_map(n),
            |m| {
                for i in 0..n as u64 {
                    m.remove(black_box(&i));
                }
                for i in 0..n as u64 {
                    m.insert(black_box(i), i);
                }
                black_box(m.len());
            },
            criterion::BatchSize::SmallInput,
        );
    });
    g.finish();
}

/// `modify` — mutate the value of an existing key in place (requires `&mut self`).
fn map_modify(c: &mut Criterion) {
    let n = 10_000usize;
    let mut g = c.benchmark_group("anchormap/modify");
    g.throughput(Throughput::Elements(n as u64));
    // Hit: key present — finds the slot, applies closure.
    g.bench_function("hit_10k", |b| {
        let mut m = filled_map(n);
        b.iter(|| {
            for i in 0..n as u64 {
                m.modify(black_box(&i), |v| *v = v.wrapping_add(1));
            }
        });
    });
    // Miss: key absent — exhausts the probe, returns false.
    g.bench_function("miss_10k", |b| {
        let mut m = filled_map(n);
        b.iter(|| {
            let mut found = 0usize;
            for i in n as u64..(2 * n) as u64 {
                if m.modify(black_box(&i), |v| *v = v.wrapping_add(1)) {
                    found += 1;
                }
            }
            black_box(found);
        });
    });
    g.finish();
}

fn map_retain(c: &mut Criterion) {
    let n = 1_000usize;
    c.bench_function("anchormap/retain_half_1k", |b| {
        b.iter_batched(
            || filled_map(n),
            |mut m| {
                m.retain(|k, _v| k % 2 == 0);
                black_box(m.len());
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn map_clear(c: &mut Criterion) {
    let n = 10_000usize;
    c.bench_function("anchormap/clear_10k", |b| {
        b.iter_batched(
            || filled_map(n),
            |mut m| {
                m.clear();
                black_box(m.len());
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn map_drain(c: &mut Criterion) {
    let n = 1_000usize;
    let mut g = c.benchmark_group("anchormap/drain");
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("1k", |b| {
        b.iter_batched(
            || filled_map(n),
            |mut m| {
                let mut sum = 0u64;
                for (k, v) in m.drain() {
                    sum = sum.wrapping_add(k).wrapping_add(v);
                }
                black_box(sum);
            },
            criterion::BatchSize::SmallInput,
        );
    });
    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Iteration / entry
// ─────────────────────────────────────────────────────────────────────────────

fn map_iter(c: &mut Criterion) {
    let n = 10_000usize;
    let m = filled_map(n);
    let mut g = c.benchmark_group("anchormap/iter");
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("10k", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for (k, v) in m.iter() {
                sum = sum.wrapping_add(*k).wrapping_add(*v);
            }
            black_box(sum);
        });
    });
    g.finish();
}

fn map_entry_or_insert(c: &mut Criterion) {
    c.bench_function("anchormap/entry_or_insert_1k", |b| {
        b.iter(|| {
            let mut m = HashMap::<u64, u64>::new(2048);
            for i in 0..1_000u64 {
                m.entry(black_box(i)).or_insert(i);
            }
        });
    });
}

/// `get_or_insert` — concurrent find-or-create, returning `&V`.
///
/// Two variants:
/// - all_present: every key exists; exercises the fast return path (no insert).
/// - all_absent: no key exists; exercises the insert path.
fn map_get_or_insert(c: &mut Criterion) {
    let n = 10_000usize;
    let mut g = c.benchmark_group("anchormap/get_or_insert");
    g.throughput(Throughput::Elements(n as u64));

    g.bench_function("all_present_10k", |b| {
        let m = filled_map(n);
        b.iter(|| {
            let mut sum = 0u64;
            for i in 0..n as u64 {
                sum = sum.wrapping_add(*m.get_or_insert(black_box(i), i * 99));
            }
            black_box(sum);
        });
    });

    g.bench_function("all_absent_10k", |b| {
        b.iter(|| {
            let m = HashMap::<u64, u64>::new(n);
            for i in 0..n as u64 {
                black_box(m.get_or_insert(black_box(i), i));
            }
        });
    });

    g.finish();
}

/// `get_or_insert_with` — lazy concurrent find-or-create.
///
/// Same two variants as `get_or_insert`; the closure is only called when the
/// key is absent so the all_present path exercises the fast-return branch.
fn map_get_or_insert_with(c: &mut Criterion) {
    let n = 10_000usize;
    let mut g = c.benchmark_group("anchormap/get_or_insert_with");
    g.throughput(Throughput::Elements(n as u64));

    g.bench_function("all_present_10k", |b| {
        let m = filled_map(n);
        b.iter(|| {
            let mut sum = 0u64;
            for i in 0..n as u64 {
                sum = sum.wrapping_add(*m.get_or_insert_with(black_box(i), || i * 99));
            }
            black_box(sum);
        });
    });

    g.bench_function("all_absent_10k", |b| {
        b.iter(|| {
            let m = HashMap::<u64, u64>::new(n);
            for i in 0..n as u64 {
                black_box(m.get_or_insert_with(black_box(i), || i));
            }
        });
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Criterion main
// ─────────────────────────────────────────────────────────────────────────────

criterion_group!(
    name = anchormap;
    config = Criterion::default().sample_size(20);
    targets =
        // Insert
        map_insert,
        map_insert_growing,
        map_insert_dup,
        map_concurrent_insert,
        // Get
        map_get_hit,
        map_get_miss,
        map_get_concurrent,
        map_contains_key,
        map_get_key_value,
        // Mutation
        map_remove,
        map_remove_reinsert,
        map_modify,
        map_retain,
        map_clear,
        map_drain,
        // Iteration / entry
        map_iter,
        map_entry_or_insert,
        map_get_or_insert,
        map_get_or_insert_with,
);

criterion_main!(anchormap);
