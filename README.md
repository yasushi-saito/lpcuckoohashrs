Lehman-Panigrahy Cuckoo hash table.

Cuckoo hash can look up an element with at most two probes, guaranteed.  On
hash conflict during insertion, elements are moved around to make room for
the new element, hence the name cuckoo.

The basic algorithm is described in the following paper:
  3.5-way Cockoo Hashing for the price of 2-and-a-bit.  Eric Lehman and Rita
  Panigrahy, European Symp. on Algorithms, 2009.

  https://pdfs.semanticscholar.org/aa7f/47954647604107fd5e67fa8162c7a785de71.pdf

It also incorporates some of the ideas presented in the following paper:

  Algorithmic Improvements for Fast Concurrent Cuckoo Hashing, Xiaozhou Li,
  David G. Andersen, Michael Kaminsky, Michael J. Freedman, Eurosys 14.

https://www.cs.princeton.edu/~mfreed/docs/cuckoo-eurosys14.pdf

# Examples

```
use lpcuckoo::LpCuckooHashMap;

let mut map = LpCuckooHashMap<String, String>::new();

map.insert("key0".to_string(), "val0".to_string());
map.insert("key1".to_string(), "val1".to_string());
assert_eq!(map.get("key0"), Some("val0".to_string()));
assert_eq!(map.remove_entry("key0"), Some(("key0".to_string(), "val0".to_string())));
assert_eq!(map.get("key0"), None);
```

# Benchmarks

On Core i5 5300U 2.3GHz, using seahash:

 - insert one entry: 100ns
 - insert 16 entries: 518ns
 - insert 128 entries: 22us
 - lookup one entry: 15ns
