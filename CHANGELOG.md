# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- Randomized stress and soak tests (`tests/stress.rs`, the third
  verification leg alongside the Loom models and Miri):
  - `distinct_key_ownership`: disjoint per-thread key ranges give every
    operation a deterministic oracle; exercises concurrent growth and
    heavy cross-stripe write traffic.
  - `shared_key_linearizability`: random get/insert/remove on a shared
    key space from many threads; each result is recorded with a
    real-time interval and every key's history is checked for
    linearizability against a sequential model by backtracking search
    (Wing–Gong style).
  - `held_reference_stability`: readers hold `&V` across unrelated
    concurrent churn (including removal of the referenced key) and
    verify the value never changes — the anchored-slot invariant under
    load.
  - `long_soak` (ignored by default): the heavy variant, run in CI.

## [0.3.0] - 2026-10-25

### Security / soundness fixes

- **Fixed a use-after-free in the guard-free `get()` API.** In previous
  versions, a concurrent `insert()` could reuse a slot tombstoned by
  `remove()`, dropping the old value while `&V` references obtained from
  `get()`, `iter()`, `get_key_value()`, or `get_or_insert*()` were still
  held. Confirmed as undefined behavior under Miri. Under `&self` access a
  slot's data is now written at most once and never dropped or overwritten;
  tombstoned data is reclaimed only by exclusive compaction (`clear`,
  `retain`, `shrink_to_fit`) or when the map is dropped.
- **Fixed data races on slot data.** `probe_entry`, `probe_for_insert`, and
  `remove`'s in-lock verify previously dereferenced `(K, V)` after only a
  Relaxed load of the metadata byte, with no happens-before edge to the
  writing stripe (the stripe lock does not order slots written from other
  stripes). All sites now use a Relaxed-then-Acquire double load on
  fingerprint matches.

### Changed

- Tombstones are never reclaimed while the map is alive through `&self`
  access: shared-path inserts claim only never-written (`META_EMPTY`) slots.
  Remove-heavy workloads on shared maps therefore accumulate dead slots
  until an exclusive compaction. (0.2.x appeared to reclaim this memory via
  slot reuse, but did so unsoundly.)
- Under `cfg(miri)` (set automatically by `cargo miri`) the internal stripe
  and growth locks use `std::sync::Mutex` instead of `parking_lot`, whose
  libc futex call is rejected by recent Miri nightlies. Normal builds are
  unchanged.

### Added

- Loom exhaustive interleaving models for the claim/publish sequence,
  cross-stripe CAS contention, concurrent segment growth and its publication
  chain, and remove's tombstone transition:

  ```text
  RUSTFLAGS="--cfg loom" cargo test --lib loom_tests
  ```

  Under `cfg(loom)` the crate's atomics and mutexes are swapped for Loom's
  simulated implementations and every feasible interleaving of the modeled
  scenarios is enumerated against the C++/Rust memory model.
- Regression tests: a `&V` held across `remove` plus 1000 concurrent inserts
  is never dropped; `clear` and `retain` reclaim tombstoned data.
- CI: clippy (`-D warnings`), the serde feature test run, the Loom suite,
  and a nightly Miri job running the full test suite and doc tests
  (`.github/workflows/rust.yml`).
- Documented MSRV (`rust-version = "1.87"`).
