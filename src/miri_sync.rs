//! Miri-only lock shim.
//!
//! parking_lot's Linux thread parker invokes `syscall(SYS_futex, ...)`
//! through libc, passing a `&AtomicI32` where recent Miri nightlies require
//! a raw `*mut u32` — a hard UB error under the interpreter (older nightlies
//! only warned). std's locks are fully supported by Miri, so under
//! `cfg(miri)` (set automatically by cargo-miri) the stripe and growth
//! locks use `std::sync::Mutex` behind a parking_lot-compatible wrapper.
//! Normal builds and Loom builds are unaffected.

/// parking_lot-compatible wrapper around `std::sync::Mutex`.
pub(crate) struct Mutex<T>(std::sync::Mutex<T>);

impl<T> Mutex<T> {
    pub(crate) fn new(value: T) -> Self {
        Self(std::sync::Mutex::new(value))
    }

    pub(crate) fn lock(&self) -> std::sync::MutexGuard<'_, T> {
        // Poisoning is not a failure mode for the map's internal invariants;
        // recover the lock, matching parking_lot (which has no poisoning).
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}
