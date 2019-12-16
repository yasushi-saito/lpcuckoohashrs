//! Lehman-Panigrahy Cuckoo hash table.
//!
//! Cuckoo hash can look up an element with at most two probes, guaranteed.  On
//! hash conflict during insertion, elements are moved around to make room for
//! the new element, hence the name cuckoo.
//!
//! The basic algorithm is described in the following paper:
//!   3.5-way Cockoo Hashing for the price of 2-and-a-bit.  Eric Lehman and Rita
//!   Panigrahy, European Symp. on Algorithms, 2009.
//!
//!   https://pdfs.semanticscholar.org/aa7f/47954647604107fd5e67fa8162c7a785de71.pdf
//!
//! It also incorporates some of the ideas presented in the following paper:
//!
//!   Algorithmic Improvements for Fast Concurrent Cuckoo Hashing, Xiaozhou Li,
//!   David G. Andersen, Michael Kaminsky, Michael J. Freedman, Eurosys 14.
//!
//! https://www.cs.princeton.edu/~mfreed/docs/cuckoo-eurosys14.pdf
//!
use std::borrow::Borrow;
use std::collections;
use std::hash::{BuildHasher, Hash, Hasher};
use std::iter::FusedIterator;
use std::mem;
use std::ptr;
use std::rc::Rc;

type RandomState = collections::hash_map::RandomState;

/// NUM_SHARDS defines the number of underlying raw hash tables.  If its value
/// is two, for example, then lookup is performed by looking up at most bucket
/// from each shard.
const NUM_SHARDS: usize = 2;
const INITIAL_NUM_NODES_PER_SHARD: usize = 16;
const BUCKET_WIDTH: usize = 2;
const LOAD_FACTOR: f64 = 0.9;

struct Node<K, V> {
    key: K,
    val: V,
}

pub struct LpCuckooHashMap<K, V, S = RandomState> {
    hash_builder: Rc<S>,
    raw: RawTable<K, V, S>,
}

// RawTable implements a fixed-capacity cuckoo hash table.
struct RawTable<K, V, S> {
    // The table is a sequence of NUM_SHARDS shards.
    // table[0..table.len()/NUM_SHARDS] is the first shard,
    // table[table.len()/NUM_SHARDS..2*table.len()/NUM_SHARDS] is the 2nd shard,
    // so on.  The table size is fixed on construction.
    //
    // INVARIANT: table.len() % NUM_SHARDS==0.
    table: Vec<Option<Node<K, V>>>,
    // Number of elements in the table.
    len: usize,
    // shard_len = table.len() / NUM_SHARDS
    shard_len: usize,
    // shard_len_mask = shard_len - 1 (note: shard_len is always a power of two)
    shard_len_mask: usize,
    // shard_len_log2 = log2(shard_len)
    shard_len_log2: usize,
    // For generating a key hasher.
    hash_builder: Rc<S>,
    // tmp_queue is used during insert, to remeber elements being visited as
    // move candidates.
    tmp_queue: Vec<Coord>,
    tmp_chain: Vec<Coord>,
}

impl<K, V> LpCuckooHashMap<K, V, RandomState>
where
    K: Eq + Hash,
{
    /// New creates an empty map with the default capacity.
    pub fn new() -> LpCuckooHashMap<K, V, RandomState> {
        Default::default()
    }
}

impl<K, V, S> LpCuckooHashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    /// Creates an empty map with the given hash builder, with the default
    /// capacity.
    pub fn with_hasher(b: S) -> LpCuckooHashMap<K, V, S> {
        let hash_builder = Rc::new(b);
        LpCuckooHashMap {
            hash_builder: hash_builder.clone(),
            raw: RawTable::with_shard_size(hash_builder.clone(), 0),
        }
    }
}

// Used in Coord::parent to indicate that the entry is a root.
const NO_PARENT: usize = 10000000;

#[derive(Copy, Clone)]
struct Coord {
    parent: usize,
    table_index: usize,
}

// Compute log2 of the value. The arg must be a power of two.
fn log2(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut l2 = 1 as usize;
    for _ in 0..64 {
        if n == (1 << l2) {
            return l2;
        }
        l2 += 1
    }
    panic!("usize");
}

impl<K, V, S> Default for LpCuckooHashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher + Default,
{
    /// Creates an empty `LpCuckooHashMap<K, V, S>`, with the `Default` value for the hasher.
    fn default() -> LpCuckooHashMap<K, V, S> {
        LpCuckooHashMap::with_hasher(Default::default())
    }
}

pub struct Iter<'a, K, V, S> {
    raw: &'a RawTable<K, V, S>,
    table_index: usize,
}

impl<'a, K, V, S> FusedIterator for Iter<'a, K, V, S> {}

impl<'a, K, V, S> Iterator for Iter<'a, K, V, S> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        while self.table_index < self.raw.table.len() {
            let slot = &self.raw.table[self.table_index];
            self.table_index += 1;
            if let Some(node) = slot {
                return Some((&node.key, &node.val));
            }
        }
        return None;
    }
}

// Drain is an iterator created by drain() method.
pub struct Drain<'a, K, V, S> {
    raw: RawTable<K, V, S>,
    orig: &'a mut RawTable<K, V, S>,
    table_index: usize,
}

impl<'a, K, V, S> FusedIterator for Drain<'a, K, V, S> {}

impl<'a, K, V, S> Drop for Drain<'a, K, V, S> {
    fn drop(&mut self) {
        while self.table_index < self.raw.table.len() {
            self.raw.table[self.table_index].take();
            self.table_index += 1;
        }
        self.raw.len = 0;
        mem::swap(self.orig, &mut self.raw)
    }
}

impl<'a, K, V, S> Iterator for Drain<'a, K, V, S> {
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        while self.table_index < self.raw.table.len() {
            let slot = &mut self.raw.table[self.table_index];
            self.table_index += 1;
            if let Some(_) = slot {
                let node = slot.take().unwrap();
                return Some((node.key, node.val));
            }
        }
        return None;
    }
}

impl<K, V, S> LpCuckooHashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    /// Looks up the map for the given key. The argument can be any type that
    /// can be borrowed as as the key type. For example, if the key is a
    /// `String`, then the arg can be a `&str`.
    pub fn get<Q>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if self.raw.table.len() == 0 {
            return None;
        }
        self.raw.get(k)
    }

    /// Returns the number of elements in the map.
    pub fn len(&self) -> usize {
        return self.raw.len;
    }

    /// Returns the number of elements that can be stored in the map without
    /// reallocation. The returned value is just a hint.
    pub fn capacity(&self) -> usize {
        return (self.raw.table.len() as f64 * LOAD_FACTOR) as usize;
    }

    /// Checks if the map contains no elements.
    pub fn is_empty(&self) -> bool {
        return self.raw.len == 0;
    }

    /// Returns an iterator that lists elements in the map, in no particular
    /// order.
    pub fn iter(&self) -> Iter<'_, K, V, S> {
        return Iter {
            raw: &self.raw,
            table_index: 0,
        };
    }

    /// Clears the map, returning all key-value pairs as an iterator. Keeps the
    /// allocated memory for reuse.
    pub fn drain(&mut self) -> Drain<'_, K, V, S> {
        let mut d = Drain {
            raw: RawTable::with_shard_size(self.hash_builder.clone(), 0),
            orig: &mut self.raw,
            table_index: 0,
        };
        mem::swap(&mut d.raw, d.orig);
        d
    }

    /// Inserts the given key/value pair in the map. If the key does not already
    /// exist in the map, this function returns None. Else, it overwrites the
    /// entry with the new value and returns the old value.
    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        if self.raw.table.len() == 0 {
            self.raw =
                RawTable::with_shard_size(self.hash_builder.clone(), INITIAL_NUM_NODES_PER_SHARD);
        }
        loop {
            match self.raw.entry(&k) {
                RawEntry::Entry(Some(node)) => return Some(mem::replace(&mut node.val, v)),
                RawEntry::Entry(ent) => {
                    *ent = Some(Node { key: k, val: v });
                    return None;
                }
                RawEntry::TableFull => (),
            }
            let org_size = self.raw.shard_len;
            let mut new_size = org_size * 2;
            loop {
                let mut new_table = RawTable::with_shard_size(self.hash_builder.clone(), new_size);
                if self.raw.into_another_table(&mut new_table) {
                    mem::replace(&mut self.raw, new_table);
                    break; // common case
                }
                undo_moves(&mut self.raw, &mut new_table);
                new_size *= 2;
                assert!(new_size < org_size * 32);
            }
        }
    }
}

enum RawEntry<'a, K, V> {
    Entry(&'a mut Option<Node<K, V>>),
    TableFull,
}

impl<K, V, S> RawTable<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    fn with_shard_size(hash_builder: Rc<S>, shard_len: usize) -> RawTable<K, V, S> {
        let table = RawTable {
            hash_builder: hash_builder,
            table: (0..shard_len * NUM_SHARDS).map(|_| None).collect(),
            len: 0,
            shard_len: shard_len,
            shard_len_mask: if shard_len == 0 { 0 } else { shard_len - 1 },
            shard_len_log2: log2(shard_len),
            tmp_chain: Default::default(),
            tmp_queue: Default::default(),
        };
        table
    }

    fn get<Q>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        for hi in 0..NUM_SHARDS {
            let mut ti = (self.make_hash(hi, k) as usize) & self.shard_len_mask;
            for _ in 0..BUCKET_WIDTH {
                match self.slot(hi, ti) {
                    Some(node) if k.eq(node.key.borrow()) => return Some(&node.val),
                    _ => (),
                }
                ti = next_node(ti, self.shard_len);
            }
        }
        return None;
    }

    fn entry(&mut self, k: &K) -> RawEntry<K, V> {
        let mut hashes: [usize; NUM_SHARDS] = Default::default();
        for hi in 0..NUM_SHARDS {
            let hash = self.make_hash(hi, k) as usize;
            hashes[hi] = hash;
            // let shard = &mut self.shards[hi];
            let mut ti = hash & self.shard_len_mask;
            let mut empty_slot: isize = -1;
            for _ in 0..BUCKET_WIDTH {
                match self.slot_mut(hi, ti) {
                    Some(node) if k.eq(node.key.borrow()) => {
                        return RawEntry::Entry(self.slot_mut(hi, ti));
                    }
                    None => {
                        if empty_slot < 0 {
                            empty_slot = ti as isize
                        }
                    }
                    _ => (),
                }
                ti = next_node(ti, self.shard_len)
            }
            if empty_slot >= 0 {
                self.len += 1;
                return RawEntry::Entry(self.slot_mut(hi, empty_slot as usize));
            }
        }

        // All slots are full.
        self.tmp_queue.clear();
        for hi in 0..NUM_SHARDS {
            let mut ti = hashes[hi] & self.shard_len_mask;
            for _ in 0..BUCKET_WIDTH {
                self.tmp_queue.push(Coord {
                    parent: NO_PARENT,
                    table_index: hi * self.shard_len + ti,
                });
                ti = next_node(ti, self.shard_len);
            }
        }
        let mut qi: usize = 0;
        while qi < self.shard_len * 2 {
            let c = self.tmp_queue[qi];
            let elem = self.table[c.table_index].as_ref().unwrap();
            for hi2 in 0..NUM_SHARDS {
                if hi2 == c.table_index >> self.shard_len_log2 {
                    continue;
                }
                //let shard = &self.shards[hi2];
                let hash = self.make_hash(hi2, &elem.key) as usize;
                let mut ti = hash & self.shard_len_mask;
                for _ in 0..BUCKET_WIDTH {
                    let c2 = Coord {
                        parent: qi,
                        table_index: hi2 * self.shard_len + ti,
                    };
                    let dest_elem = &self.table[c2.table_index];
                    if let None = dest_elem {
                        let vacated = self.evict_chain(c2);
                        self.len += 1;
                        return RawEntry::Entry(&mut self.table[vacated.table_index]);
                    }
                    self.tmp_queue.push(c2);
                    ti = next_node(ti, self.shard_len);
                }
            }
            qi += 1;
        }
        return RawEntry::TableFull;
    }

    fn into_another_table(&mut self, dest: &mut RawTable<K, V, S>) -> bool {
        for node in self.table.iter_mut() {
            let node = node.take();
            if let Some(node) = node {
                match dest.entry(&node.key) {
                    RawEntry::Entry(Some(_node)) => panic!("double insert"),
                    RawEntry::Entry(ent) => *ent = Some(node),
                    RawEntry::TableFull => return false,
                }
            }
        }
        return true;
    }

    // Compute the hash value for the given table shard (hi) and key.
    #[inline]
    fn make_hash<Q>(&self, hi: usize, k: &Q) -> u64
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let mut state = self.hash_builder.build_hasher();
        hi.hash(&mut state);
        k.hash(&mut state);
        state.finish()
    }

    fn evict_chain(&mut self, mut tail: Coord) -> Coord {
        let queue = &self.tmp_queue;
        unsafe {
            let tmp_chain: *mut Vec<Coord> = &mut self.tmp_chain;
            (*tmp_chain).clear();
            (*tmp_chain).push(tail);
            while tail.parent != NO_PARENT {
                assert!(tail.parent < queue.len());
                (*tmp_chain).push(queue[tail.parent]);
                tail = queue[tail.parent];
            }
            assert!((*tmp_chain).len() >= 2);
            for i in 0..(*tmp_chain).len() - 1 {
                let c0: Coord = (*tmp_chain)[i];
                let c1: Coord = (*tmp_chain)[i + 1];
                let v0 = &mut self.table[c0.table_index] as *mut Option<Node<K, V>>;
                let v1 = &mut self.table[c1.table_index] as *mut Option<Node<K, V>>;
                ptr::swap(v0, v1);
            }
            let vacated = (*tmp_chain)[(*tmp_chain).len() - 1];
            if let Some(_) = self.table[vacated.table_index] {
                panic!("blah");
            }
            return vacated;
        }
    }

    fn slot(&self, hi: usize, ti: usize) -> &Option<Node<K, V>> {
        return &self.table[(hi << self.shard_len_log2) + ti];
    }

    fn slot_mut(&mut self, hi: usize, ti: usize) -> &mut Option<Node<K, V>> {
        return &mut self.table[(hi << self.shard_len_log2) + ti];
    }
}

fn undo_moves<K, V, S>(orgtable: &mut RawTable<K, V, S>, newtable: &mut RawTable<K, V, S>) {
    println!("undoing inserts!");
    //panic!("blah");
    let mut last_ti = 0;

    let mut try_insert = |k: K, v: V| loop {
        let node = &mut orgtable.table[last_ti];
        if let None = *node {
            *node = Some(Node { key: k, val: v });
            return;
        }
        last_ti += 1;
        if last_ti >= orgtable.table.len() {
            panic!("blah")
        }
    };

    for node in newtable.table.iter_mut() {
        let node = node.take();
        if let Some(node) = node {
            try_insert(node.key, node.val)
        }
    }
}

#[inline]
fn next_node(ti: usize, max_size: usize) -> usize {
    let next = ti + 1;
    if next >= max_size {
        return 0;
    }
    return next;
}

#[cfg(test)]
mod tests {
    use fasthash::xx::Hash64;
    use std::collections::HashSet;
    type Key = String;
    type Value = String;
    type Map = crate::LpCuckooHashMap<Key, Value, Hash64>;

    fn new_map() -> Map {
        Map::with_hasher(Hash64)
    }

    #[test]
    fn simple() {
        //let mut m = Map::new();
        let mut m = new_map();
        assert_eq!(m.len(), 0);
        assert_eq!(m.capacity(), 0);
        assert_eq!(m.get("blah"), None);
        assert_eq!(m.insert(String::from("blah"), String::from("foo")), None);
        assert_eq!(m.len(), 1);
        assert_eq!(m.get("blah"), Some(&String::from("foo")));
        assert_eq!(m.get("blah2"), None);
    }

    #[test]
    fn resize() {
        let key_fn = |i: usize| (i + 100).to_string();
        let val_fn = |i: usize| (i + 200).to_string();

        const NUM_ELEMS: usize = crate::INITIAL_NUM_NODES_PER_SHARD * crate::NUM_SHARDS + 1;
        let mut m = new_map();
        for i in 0..NUM_ELEMS {
            let key = key_fn(i);
            let val = val_fn(i);
            println!("{}: Insert {}->{}", i, key, val);
            assert_eq!(m.insert(key, val), None);
            assert_eq!(m.len(), i + 1);
        }
        for i in 0..NUM_ELEMS {
            let key = key_fn(i);
            let want = val_fn(i);
            println!("{}: Get {}", i, key);
            assert_eq!(m.get(&key), Some(&want));
        }
    }

    #[test]
    fn iter() {
        let key_fn = |i: usize| (i + 100).to_string();
        let val_fn = |i: usize| (i + 200).to_string();
        let mut m = new_map();
        const NUM_ELEMS: usize = 1000;
        for i in 0..NUM_ELEMS {
            let key = key_fn(i);
            let val = val_fn(i);
            assert_eq!(m.insert(key, val), None);

            let mut got_keys = HashSet::<Key>::new();
            for (key, val) in m.iter() {
                assert!(got_keys.insert(key.clone()));
                let key_val = key.parse::<i32>().unwrap();
                assert!(key_val >= 100 && key_val < 100 + NUM_ELEMS as i32);
                assert_eq!(*val, (key_val + 100).to_string());
            }
            assert_eq!(got_keys.len(), i + 1);
        }
    }

    #[test]
    fn drain() {
        let key_fn = |i: usize| (i + 100).to_string();
        let val_fn = |i: usize| (i + 200).to_string();
        let mut m = new_map();
        const NUM_ELEMS: usize = 1000;
        for _rep in 0..2 {
            for i in 0..NUM_ELEMS {
                let key = key_fn(i);
                let val = val_fn(i);
                assert_eq!(m.insert(key, val), None);
            }
            let mut got_keys = HashSet::<Key>::new();
            for (key, val) in m.drain() {
                assert!(got_keys.insert(key.clone()));
                let key_val = key.parse::<i32>().unwrap();
                assert!(key_val >= 100 && key_val < 100 + NUM_ELEMS as i32);
                assert_eq!(*val, (key_val + 100).to_string());
            }
            assert_eq!(got_keys.len(), NUM_ELEMS);
            assert_eq!(m.len(), 0);
            assert!(m.capacity() >= 500);
        }
    }
}
