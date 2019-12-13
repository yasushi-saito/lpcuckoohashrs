use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time;

use fasthash::sea::Hash64;
use lpcuckoo::LpCuckooHashMap;

pub fn criterion_benchmark(c: &mut Criterion) {
    type Map = crate::LpCuckooHashMap<i32, i32, Hash64>;
    fn new_map() -> Map {
        Map::with_hasher(Hash64)
    }
    c.bench_function("lpcuckoo new", |b| {
        b.iter(|| {
            let _x = new_map();
        })
    });

    let insert_bench = |elems, iters| {
        let mut x = new_map();
        if iters <= 0 {
            panic!("iters")
        }
        let timer = time::Instant::now();
        for _ in 0..iters {
            x = new_map();
            for key in 0..elems {
                x.insert(black_box(key), key + 10);
            }
        }
        let elapsed = timer.elapsed();
        for key in 0..elems {
            match x.get(&key) {
                Some(&val) if val == key + 10 => (),
                _ => panic!("incorrect"),
            }
        }
        elapsed
    };

    //let elems_list = [2, 8, 64, 128, 1024];
    let elems_list: Vec<i32> = vec![1, 16, 128, 1024];
    let mut group = c.benchmark_group("lpcuckoo insert");
    for elems in elems_list.iter() {
        group.bench_function(format!("{}", *elems), |b| {
            b.iter_custom(|iters| insert_bench(*elems, iters))
        });
    }
    group.finish();

    let lookup_bench = |elems, iters| {
        let mut x = new_map();
        for key in 0..elems {
            x.insert(black_box(key), key + 10);
        }
        let timer = time::Instant::now();
        for _ in 0..iters {
            for key in 0..elems {
                match x.get(black_box(&key)) {
                    Some(&val) if val == key + 10 => (),
                    _ => panic!("incorrect"),
                }
            }
        }
        timer.elapsed()
    };
    let mut group = c.benchmark_group("lpcuckoo lookup");
    for elems in elems_list.iter() {
        group.bench_function(format!("{}", *elems), |b| {
            b.iter_custom(|iters| lookup_bench(*elems, iters))
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
