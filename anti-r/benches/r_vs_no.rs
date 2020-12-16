use criterion::AxisScale;
use criterion::PlotConfiguration;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

mod rbench;

use rbench::build_rtree;
use rbench::gen_points;
use rbench::gen_query;
use rbench::mutate;
use rbench::preprocess_points;
use rbench::rtree_gen_update;
use rbench::Loc;

use anti_r::SpatVec;

criterion_group!(benches, load, query, update);
criterion_main!(benches);

fn plotconf() -> PlotConfiguration {
    PlotConfiguration::default().summary_scale(AxisScale::Logarithmic)
}

fn load(c: &mut Criterion) {
    let mut group = c.benchmark_group("load");
    group.plot_config(plotconf());
    group.warm_up_time(std::time::Duration::from_millis(200));
    group.measurement_time(std::time::Duration::from_secs(1));
    group.sample_size(50);
    for s in 3..11 {
        let num = 1 << (s * 2);
        let points = gen_points(num);
        // just cause rtree happens to not use the exact "on disk" data format does not get penalized
        // here
        let data = preprocess_points(&points);
        group.bench_with_input(BenchmarkId::new("rtree", num), &num, |b, _num| {
            b.iter_with_setup(|| data.clone(), |data| build_rtree(data))
        });
        group.bench_with_input(BenchmarkId::new("antir", num), &num, |b, _num| {
            b.iter_with_setup(|| points.clone(), |points| SpatVec::new_from(points))
        });
    }

    group.finish();
}

fn query(c: &mut Criterion) {
    let mut group = c.benchmark_group("query");
    group.plot_config(plotconf());
    group.warm_up_time(std::time::Duration::from_millis(200));
    group.measurement_time(std::time::Duration::from_secs(1));
    group.sample_size(250);
    for s in 6..21 {
        let num = 1 << s;
        let points = gen_points(num);
        let data = preprocess_points(&points);
        let tree = build_rtree(data);
        let mut rng = rand::thread_rng();

        group.bench_with_input(BenchmarkId::new("rtree", num), &num, |b, _num| {
            b.iter_with_setup(
                || {
                    let (lu, rd) = gen_query(&mut rng);
                    let aabb = rstar::AABB::from_corners(lu, rd);
                    aabb
                },
                |query| {
                    let res = tree
                        .locate_in_envelope(&query)
                        // gotta do something with the iterator, otherwise its lazy
                        .map(|p| p.data)
                        .sum::<usize>();
                    res
                },
            )
        });

        let points = SpatVec::new_from(points);

        group.bench_with_input(BenchmarkId::new("antir", num), &num, |b, _num| {
            b.iter_with_setup(
                || gen_query(&mut rng),
                |(lu, rd)| {
                    let slice = points.as_spat_slice();
                    let (start, end) = slice.query_aabb(&lu, &rd);
                    // this is to be fair towards rtree, which returns an iterator that needs to be consumed
                    slice[start..end].iter().map(|p| p.1).sum::<usize>()
                },
            )
        });
    }

    group.finish()
}

fn update(c: &mut Criterion) {
    let mut group = c.benchmark_group("update");

    group.plot_config(plotconf());
    group.warm_up_time(std::time::Duration::from_millis(200));
    group.measurement_time(std::time::Duration::from_secs(1));
    group.sample_size(50);
    for s in 3..11 {
        let num = 1 << (s * 2);
        let points = gen_points(num);
        let data = preprocess_points(&points);
        let tree = build_rtree(data);

        let updates = rtree_gen_update(&tree);
        group.bench_with_input(BenchmarkId::new("rtree", num), &num, |b, _num| {
            b.iter_with_setup(
                || tree.clone(),
                |mut tree| {
                    for (pre, post) in updates.iter() {
                        let d = tree.remove_at_point(pre).unwrap().data;
                        tree.insert(Loc::new(d, *post));
                    }
                    tree
                },
            )
        });

        let mut rng = rand::thread_rng();

        let updates = points
            .iter()
            .map(|(p, _d)| p)
            .map(|p| mutate(&mut rng, *p))
            .collect::<Vec<_>>();

        let points = SpatVec::new_from(points);

        group.bench_with_input(BenchmarkId::new("antir", num), &num, |b, _num| {
            b.iter_with_setup(
                || (points.clone(), updates.iter()),
                |(mut points, mut updates)| {
                    points.map(|(k, _v)| {
                        let u = updates.next().unwrap();
                        *k = *u;
                    });
                    points
                },
            )
        });
    }
}
