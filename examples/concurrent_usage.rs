// Concurrent usage example for HashMap.
//
// Demonstrates lock-free reads from multiple threads while a writer inserts
// entries, exploiting the elastic hashing property that elements never move
// after insertion.
//
// Run with:
//   cargo run --example concurrent_usage

use std::sync::Arc;
use anchormap::HashMap;

fn main() {
    // ── 1. Basic concurrent reads ─────────────────────────────────────────────
    // Pre-populate the map, then spawn 8 reader threads.
    // No locking is needed on the read side.
    println!("── Concurrent reads ──");

    let map = Arc::new(HashMap::<u32, u32>::new(512));
    for i in 0..200u32 {
        map.insert(i, i * 10);
    }

    std::thread::scope(|s| {
        for t in 0..8 {
            let map = Arc::clone(&map);
            s.spawn(move || {
                let mut found = 0usize;
                for i in 0..200u32 {
                    if map.get(&i) == Some(&(i * 10)) {
                        found += 1;
                    }
                }
                println!("  Thread {t}: found {found}/200 entries");
            });
        }
    });

    // ── 2. Concurrent writes + reads ──────────────────────────────────────────
    // Writers and readers operate simultaneously.
    // Readers may observe some or all of the inserted entries.
    println!("\n── Concurrent writes + reads ──");

    let map = Arc::new(HashMap::<u32, String>::new(1024));

    std::thread::scope(|s| {
        // 4 writer threads insert disjoint key ranges
        for t in 0..4u32 {
            let map = Arc::clone(&map);
            s.spawn(move || {
                let start = t * 50;
                for i in start..start + 50 {
                    map.insert(i, format!("value-{i}"));
                }
                println!("  Writer {t}: inserted keys {}..{}", start, start + 50);
            });
        }

        // 2 reader threads scan the key space continuously
        for r in 0..2 {
            let map = Arc::clone(&map);
            s.spawn(move || {
                let mut seen = 0usize;
                for i in 0..200u32 {
                    if map.get(&i).is_some() {
                        seen += 1;
                    }
                }
                println!("  Reader {r}: observed {seen}/200 entries (snapshot during writes)");
            });
        }
    });

    println!("\nFinal map size: {}/{}", map.len(), map.capacity());

    // ── 3. Error cases ─────────────────────────────────────────────────────────
    println!("\n── Error cases ──");

    let small = HashMap::<&str, u32>::new(4);
    println!("  insert 'a': {:?}", small.insert("a", 1));
    println!("  insert 'a' again: {:?}", small.insert("a", 2)); // KeyAlreadyInserted
    println!("  get 'a': {:?}", small.get(&"a"));               // Some(1), unchanged
}
