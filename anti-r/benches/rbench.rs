use rstar::primitives::PointWithData;
pub type Loc = PointWithData<usize, [f64; 2]>;
pub type RTree = rstar::RTree<Loc>;
use rand::Rng;

const MOVE: f64 = 0.2;

pub fn gen_points(num: usize) -> Vec<([f64; 2], usize)> {
    let mut v = Vec::with_capacity(num);
    let mut rng = rand::thread_rng();
    for i in 0..num {
        let x = rng.gen_range(-1000.0..1000.);
        let y = rng.gen_range(-1000.0..1000.);
        v.push(([x, y], i));
    }
    v
}

pub fn preprocess_points(data: &[([f64; 2], usize)]) -> Vec<Loc> {
    data.iter().map(|(p, d)| Loc::new(*d, *p)).collect()
}

pub fn build_rtree(data: Vec<Loc>) -> RTree {
    RTree::bulk_load(data)
}

pub fn gen_query<R: Rng>(mut rng: R) -> ([f64; 2], [f64; 2]) {
    let center_x = rng.gen_range(-1000.0..1000.);
    let center_y = rng.gen_range(-1000.0..1000.);
    let offset = rng.gen_range(0.0..10.);
    let lu = [center_x - offset, center_y - offset];
    let rd = [center_x + offset, center_y + offset];
    (lu, rd)
}

pub fn mutate<R: Rng>(mut r: R, p: [f64; 2]) -> [f64; 2] {
    let off_x = r.gen_range(-MOVE..MOVE);
    let off_y = r.gen_range(-MOVE..MOVE);
    [p[0] + off_x, p[1] + off_y]
}

// can't do inline updates on rtree, don't want the generation overhead inside the benchmark
// so we need to pre-gen the whole thing (or use criterion)
pub fn rtree_gen_update(rtree: &RTree) -> Vec<([f64; 2], [f64; 2])> {
    let mut rng = rand::thread_rng();
    rtree
        .iter()
        .map(|p| p.position())
        .map(|p| (*p, mutate(&mut rng, *p)))
        .collect()
}
