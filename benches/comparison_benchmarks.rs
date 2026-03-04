//! Comparative benchmarks — anchormap vs DashMap vs Papaya vs RwLock<HashMap>
//!
//! Covers the operations where concurrent map implementations differ most:
//!   - Single-threaded get and insert (baseline, no contention)
//!   - Concurrent reads (anchormap's primary strength)
//!   - Mixed read/write (realistic workload)
//!
//! Run:
//!   cargo bench --bench comparison_benchmarks
//!   cargo bench --bench comparison_benchmarks -- get_hit   # filter

use std::hint::black_box;
use std::sync::{Arc, RwLock};
use std::collections::HashMap as StdHashMap;
use std::thread;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use anchormap::HashMap as AnchorMap;
use dashmap::DashMap;
use papaya::HashMap as PapayaMap;

const N: usize = 10_000;

// ── helpers ───────────────────────────────────────────────────────────────────

fn filled_anchor(n: usize) -> AnchorMap<u64, u64> {
    let m = AnchorMap::new(n);
    for i in 0..n as u64 { m.insert(i, i * 2); }
    m
}

fn filled_dash(n: usize) -> DashMap<u64, u64> {
    let m = DashMap::with_capacity(n);
    for i in 0..n as u64 { m.insert(i, i * 2); }
    m
}

fn filled_papaya(n: usize) -> PapayaMap<u64, u64> {
    let m = PapayaMap::with_capacity(n);
    { let g = m.pin(); for i in 0..n as u64 { g.insert(i, i * 2); } }
    m
}

fn filled_rwlock(n: usize) -> RwLock<StdHashMap<u64, u64>> {
    let mut m = StdHashMap::with_capacity(n);
    for i in 0..n as u64 { m.insert(i, i * 2); }
    RwLock::new(m)
}

// ── single-threaded get ───────────────────────────────────────────────────────

fn cmp_get_hit(c: &mut Criterion) {
    let mut g = c.benchmark_group("compare/get_hit");
    g.throughput(Throughput::Elements(N as u64));

    let anchor = filled_anchor(N);
    g.bench_function("anchormap", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for i in 0..N as u64 {
                sum = sum.wrapping_add(*anchor.get(black_box(&i)).unwrap());
            }
            black_box(sum);
        });
    });

    let dash = filled_dash(N);
    g.bench_function("dashmap", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for i in 0..N as u64 {
                sum = sum.wrapping_add(*dash.get(black_box(&i)).unwrap());
            }
            black_box(sum);
        });
    });

    let papaya = filled_papaya(N);
    g.bench_function("papaya", |b| {
        b.iter(|| {
            let guard = papaya.pin();
            let mut sum = 0u64;
            for i in 0..N as u64 {
                sum = sum.wrapping_add(*guard.get(black_box(&i)).unwrap());
            }
            black_box(sum);
        });
    });

    let rwlock = filled_rwlock(N);
    g.bench_function("rwlock_std", |b| {
        b.iter(|| {
            let m = rwlock.read().unwrap();
            let mut sum = 0u64;
            for i in 0..N as u64 {
                sum = sum.wrapping_add(*m.get(black_box(&i)).unwrap());
            }
            black_box(sum);
        });
    });

    g.finish();
}

// ── concurrent reads ──────────────────────────────────────────────────────────

fn cmp_get_concurrent(c: &mut Criterion) {
    let mut g = c.benchmark_group("compare/get_concurrent");
    g.throughput(Throughput::Elements(N as u64));

    for &threads in &[2usize, 4, 8] {
        let anchor = Arc::new(filled_anchor(N));
        g.bench_with_input(BenchmarkId::new("anchormap", threads), &threads, |b, &t| {
            b.iter(|| {
                let handles: Vec<_> = (0..t).map(|_| {
                    let m = Arc::clone(&anchor);
                    thread::spawn(move || {
                        let mut sum = 0u64;
                        for i in 0..N as u64 {
                            sum = sum.wrapping_add(*m.get(black_box(&i)).unwrap());
                        }
                        black_box(sum);
                    })
                }).collect();
                for h in handles { h.join().unwrap(); }
            });
        });

        let dash = Arc::new(filled_dash(N));
        g.bench_with_input(BenchmarkId::new("dashmap", threads), &threads, |b, &t| {
            b.iter(|| {
                let handles: Vec<_> = (0..t).map(|_| {
                    let m = Arc::clone(&dash);
                    thread::spawn(move || {
                        let mut sum = 0u64;
                        for i in 0..N as u64 {
                            sum = sum.wrapping_add(*m.get(black_box(&i)).unwrap());
                        }
                        black_box(sum);
                    })
                }).collect();
                for h in handles { h.join().unwrap(); }
            });
        });

        let papaya = Arc::new(filled_papaya(N));
        g.bench_with_input(BenchmarkId::new("papaya", threads), &threads, |b, &t| {
            b.iter(|| {
                let handles: Vec<_> = (0..t).map(|_| {
                    let m = Arc::clone(&papaya);
                    // pin() is !Send — must be called inside the spawned thread.
                    thread::spawn(move || {
                        let guard = m.pin();
                        let mut sum = 0u64;
                        for i in 0..N as u64 {
                            sum = sum.wrapping_add(*guard.get(black_box(&i)).unwrap());
                        }
                        black_box(sum);
                    })
                }).collect();
                for h in handles { h.join().unwrap(); }
            });
        });

        let rwlock = Arc::new(filled_rwlock(N));
        g.bench_with_input(BenchmarkId::new("rwlock_std", threads), &threads, |b, &t| {
            b.iter(|| {
                let handles: Vec<_> = (0..t).map(|_| {
                    let m = Arc::clone(&rwlock);
                    thread::spawn(move || {
                        let map = m.read().unwrap();
                        let mut sum = 0u64;
                        for i in 0..N as u64 {
                            sum = sum.wrapping_add(*map.get(black_box(&i)).unwrap());
                        }
                        black_box(sum);
                    })
                }).collect();
                for h in handles { h.join().unwrap(); }
            });
        });
    }

    g.finish();
}

// ── single-threaded insert ────────────────────────────────────────────────────

fn cmp_insert(c: &mut Criterion) {
    let mut g = c.benchmark_group("compare/insert");
    g.throughput(Throughput::Elements(N as u64));

    g.bench_function("anchormap", |b| {
        b.iter(|| {
            let m = AnchorMap::<u64, u64>::new(N);
            for i in 0..N as u64 { m.insert(black_box(i), i); }
        });
    });

    g.bench_function("dashmap", |b| {
        b.iter(|| {
            let m = DashMap::<u64, u64>::with_capacity(N);
            for i in 0..N as u64 { m.insert(black_box(i), i); }
        });
    });

    g.bench_function("papaya", |b| {
        b.iter(|| {
            let m = PapayaMap::<u64, u64>::with_capacity(N);
            let g = m.pin();
            for i in 0..N as u64 { g.insert(black_box(i), i); }
        });
    });

    g.bench_function("rwlock_std", |b| {
        b.iter(|| {
            let m = RwLock::new(StdHashMap::<u64, u64>::with_capacity(N));
            for i in 0..N as u64 { m.write().unwrap().insert(black_box(i), i); }
        });
    });

    g.finish();
}

// ── concurrent mixed: read-heavy (7 readers, 1 writer) ───────────────────────
//
// Models a realistic cache/registry workload: most threads look up existing
// entries while one thread continuously adds new ones. Reader and writer run
// simultaneously; setup (map creation) is excluded from measurement time.

fn cmp_mixed_read_heavy(c: &mut Criterion) {
    const READERS:    usize = 7;
    const WRITE_OPS:  usize = N / 10; // 1 000 inserts per iteration
    const TOTAL_OPS:  u64   = (READERS * N + WRITE_OPS) as u64;

    let mut g = c.benchmark_group("compare/mixed_read_heavy");
    g.throughput(Throughput::Elements(TOTAL_OPS));

    g.bench_function("anchormap", |b| {
        b.iter_batched(
            || Arc::new(filled_anchor(N + WRITE_OPS)),
            |m| {
                let handles: Vec<_> = (0..READERS).map(|_| {
                    let m = Arc::clone(&m);
                    thread::spawn(move || {
                        let mut sum = 0u64;
                        for i in 0..N as u64 {
                            if let Some(v) = m.get(black_box(&i)) { sum = sum.wrapping_add(*v); }
                        }
                        black_box(sum);
                    })
                })
                .chain(std::iter::once({
                    let m = Arc::clone(&m);
                    thread::spawn(move || {
                        for i in N as u64..(N + WRITE_OPS) as u64 { m.insert(black_box(i), i); }
                    })
                }))
                .collect();
                for h in handles { h.join().unwrap(); }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    g.bench_function("dashmap", |b| {
        b.iter_batched(
            || Arc::new(filled_dash(N + WRITE_OPS)),
            |m| {
                let handles: Vec<_> = (0..READERS).map(|_| {
                    let m = Arc::clone(&m);
                    thread::spawn(move || {
                        let mut sum = 0u64;
                        for i in 0..N as u64 {
                            if let Some(v) = m.get(black_box(&i)) { sum = sum.wrapping_add(*v); }
                        }
                        black_box(sum);
                    })
                })
                .chain(std::iter::once({
                    let m = Arc::clone(&m);
                    thread::spawn(move || {
                        for i in N as u64..(N + WRITE_OPS) as u64 { m.insert(black_box(i), i); }
                    })
                }))
                .collect();
                for h in handles { h.join().unwrap(); }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    g.bench_function("papaya", |b| {
        b.iter_batched(
            || Arc::new(filled_papaya(N + WRITE_OPS)),
            |m| {
                let handles: Vec<_> = (0..READERS).map(|_| {
                    let m = Arc::clone(&m);
                    thread::spawn(move || {
                        let guard = m.pin();
                        let mut sum = 0u64;
                        for i in 0..N as u64 {
                            if let Some(v) = guard.get(black_box(&i)) { sum = sum.wrapping_add(*v); }
                        }
                        black_box(sum);
                    })
                })
                .chain(std::iter::once({
                    let m = Arc::clone(&m);
                    thread::spawn(move || {
                        let guard = m.pin();
                        for i in N as u64..(N + WRITE_OPS) as u64 { guard.insert(black_box(i), i); }
                    })
                }))
                .collect();
                for h in handles { h.join().unwrap(); }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    g.bench_function("rwlock_std", |b| {
        b.iter_batched(
            || Arc::new(filled_rwlock(N + WRITE_OPS)),
            |m| {
                let handles: Vec<_> = (0..READERS).map(|_| {
                    let m = Arc::clone(&m);
                    thread::spawn(move || {
                        let map = m.read().unwrap();
                        let mut sum = 0u64;
                        for i in 0..N as u64 {
                            if let Some(v) = map.get(black_box(&i)) { sum = sum.wrapping_add(*v); }
                        }
                        black_box(sum);
                    })
                })
                .chain(std::iter::once({
                    let m = Arc::clone(&m);
                    thread::spawn(move || {
                        let mut map = m.write().unwrap();
                        for i in N as u64..(N + WRITE_OPS) as u64 { map.insert(black_box(i), i); }
                    })
                }))
                .collect();
                for h in handles { h.join().unwrap(); }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    g.finish();
}

// ── Criterion main ────────────────────────────────────────────────────────────

criterion_group!(
    name = comparison;
    config = Criterion::default().sample_size(20);
    targets =
        cmp_get_hit,
        cmp_get_concurrent,
        cmp_insert,
        cmp_mixed_read_heavy,
);

criterion_main!(comparison);
