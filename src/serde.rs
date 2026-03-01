//! Serde support for [`HashMap`](crate::HashMap).
//!
//! Enabled with the `serde` feature flag:
//!
//! ```toml
//! [dependencies]
//! anchormap = { version = "0.1", features = ["serde"] }
//! ```

use std::hash::Hash;
use std::marker::PhantomData;

use serde::de::{MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::hashing::{AHashHasher, TableHasher};
use crate::HashMap;

impl<K, V, H> Serialize for HashMap<K, V, H>
where
    K: Eq + Hash + Serialize,
    V: Serialize,
    H: TableHasher,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.len()))?;
        for (k, v) in self.iter() {
            map.serialize_entry(k, v)?;
        }
        map.end()
    }
}

impl<'de, K, V> Deserialize<'de> for HashMap<K, V, AHashHasher>
where
    K: Eq + Hash + Deserialize<'de>,
    V: Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_map(HashMapVisitor(PhantomData))
    }
}

struct HashMapVisitor<K, V>(PhantomData<(K, V)>);

impl<'de, K, V> Visitor<'de> for HashMapVisitor<K, V>
where
    K: Eq + Hash + Deserialize<'de>,
    V: Deserialize<'de>,
{
    type Value = HashMap<K, V, AHashHasher>;

    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a map")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> Result<Self::Value, A::Error> {
        let map = HashMap::new(access.size_hint().unwrap_or(0));
        while let Some((k, v)) = access.next_entry()? {
            let _ = map.insert(k, v);
        }
        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serde_json_roundtrip() {
        let map = HashMap::<String, u32>::new(16);
        map.insert("alpha".to_string(), 1);
        map.insert("beta".to_string(), 2);
        map.insert("gamma".to_string(), 3);

        let json = serde_json::to_string(&map).unwrap();
        let restored: HashMap<String, u32> = serde_json::from_str(&json).unwrap();

        assert_eq!(map.len(), restored.len());
        assert_eq!(restored.get("alpha"), Some(&1));
        assert_eq!(restored.get("beta"), Some(&2));
        assert_eq!(restored.get("gamma"), Some(&3));
    }

    #[test]
    fn test_serde_empty_map() {
        let map = HashMap::<String, u32>::new(16);
        let json = serde_json::to_string(&map).unwrap();
        assert_eq!(json, "{}");
        let restored: HashMap<String, u32> = serde_json::from_str(&json).unwrap();
        assert!(restored.is_empty());
    }

    #[test]
    fn test_serde_duplicate_keys_first_wins() {
        // JSON technically allows duplicate keys; first-wins matches insert semantics.
        let json = r#"{"key": 1, "key": 2}"#;
        let map: HashMap<String, u32> = serde_json::from_str(json).unwrap();
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("key"), Some(&1));
    }
}
