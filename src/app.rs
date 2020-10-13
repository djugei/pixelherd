use crate::config;

use crate::blip::Blip;
use crate::brains::Brain;
use crate::stablevec::StableVec;
use anti_r::vec::SpatVec;
use anti_r::SpatSlice;
use atomic::Atomic;
use piston::input::UpdateArgs;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng as DetRng;
use rayon::prelude::IndexedParallelIterator;
use rayon::prelude::IntoParallelRefMutIterator;
use rayon::prelude::ParallelIterator;

pub type Tree = SpatVec<[f64; 2], usize>;
pub type TreeRef<'a> = SpatSlice<'a, [f64; 2], usize>;
// each step foodgrid is copied into a read only and a (synchronized) write-only part
pub type FoodGrid = [[Atomic<f64>; config::FOOD_HEIGHT]; config::FOOD_WIDTH];
// safety: always change this in sync
pub type OldFoodGrid = [[f64; config::FOOD_HEIGHT]; config::FOOD_WIDTH];

#[derive(Debug)]
pub struct App<B: Brain + Send + Clone + Sync> {
    blips: StableVec<Blip<B>>,
    foodgrid: FoodGrid,
    // todo: replace with a simple sorted list
    tree: Tree,
    time: f64,
    rng: DetRng,
}

impl<B: Brain + Send + Clone + Sync> App<B> {
    pub fn foodgrid(&self) -> &FoodGrid {
        &self.foodgrid
    }
    pub fn tree<'a>(&'a self) -> TreeRef<'a> {
        (&self.tree).into()
    }
    pub fn blips(&self) -> &StableVec<Blip<B>> {
        &self.blips
    }
    pub fn new(seed: u64) -> Self {
        let rng = DetRng::seed_from_u64(seed);
        let foodgrid = [[0.; config::FOOD_HEIGHT]; config::FOOD_WIDTH];
        //safety: this is absolutely not safe as i am relying on the internal memory layout of a third
        // party library that is almost guaranteed to not match on 32 bit platforms.
        //
        // however i see no other way to initialize this array
        // try_from is only implemented for array up to size 32 because fucking rust has no const
        // generics
        // atomics are not copy, so the [0.;times] constructor does not work
        // this is an actual value, not a reference so i need to actually change the value instead of
        // "as-casting" the pointer
        let foodgrid = unsafe { std::mem::transmute(foodgrid) };

        // Create a new game and run it.
        let mut app = App {
            blips: StableVec::with_capacity(config::INITIAL_CELLS),
            tree: SpatVec::new_from(Vec::with_capacity(config::INITIAL_CELLS)),
            foodgrid,
            time: 0.,
            rng: rng,
        };

        for _ in 0..config::INITIAL_CELLS {
            app.blips.push(Blip::new(&mut app.rng));
        }

        for w in 0..config::FOOD_WIDTH {
            for h in 0..config::FOOD_HEIGHT {
                //fixme: this should be an exponential distribution instead
                if app.rng.gen_range(0, 3) == 1 {
                    *app.foodgrid[w][h].get_mut() = app.rng.gen_range(0., 10.);
                }
            }
        }
        app
    }
    // fixme: make sure dt is used literally on every change
    pub fn update(&mut self, args: &UpdateArgs) {
        self.time += args.dt;
        // update the inputs
        // todo: don't clone here, keep two buffers and swap instead
        let mut new = self.blips.clone();

        let spawns = std::sync::Mutex::new(Vec::new());

        let iter = new
            .inner_mut()
            .par_iter_mut()
            .zip(self.blips.inner())
            .enumerate()
            //todo: maybe move this into the for_each
            // as the blips are assumed to be quite dense
            // (better chunking for parallel execution)
            .flat_map(|(index, (new, old))| {
                if let (Some(new), Some(old)) = (new, old) {
                    Some((index, (new, old)))
                } else {
                    None
                }
            });

        let mut oldgrid: OldFoodGrid = [[0.; config::FOOD_HEIGHT]; config::FOOD_WIDTH];
        for (w, r) in oldgrid.iter_mut().zip(self.foodgrid.iter_mut()) {
            for (w, r) in w.iter_mut().zip(r.iter_mut()) {
                *w = *r.get_mut();
            }
        }

        // new is write only. if you need data from the last iteration
        // get it from old only.
        iter.for_each(|(index, (new, old))| {
            let seed = self.time.to_bits();
            let seed = seed ^ (index as u64);
            let mut rng = DetRng::seed_from_u64(seed);

            let spawn = new.update(
                &mut rng,
                old,
                &self.blips,
                self.tree(),
                &oldgrid,
                &self.foodgrid,
                self.time,
                args.dt,
            );
            if let Some(spawn) = spawn {
                let mut guard = spawns.lock().unwrap();
                guard.push((index, spawn));
            }
        });

        let mut spawns = spawns.into_inner().unwrap();
        // gotta sort so insertion order is deterministic
        spawns.sort_unstable_by_key(|(i, _)| *i);

        new.extend(spawns.into_iter().map(|(_i, e)| e));

        // todo: drop some meat on death
        new.retain(|blip| blip.status.hp > 0.);

        if new.len() < config::REPLACEMENT {
            println!("force-spawned");
            new.push(Blip::new(&mut self.rng));
        }

        self.blips = new;

        // chance could be > 1 if dt or replenish are big enough
        let mut chance = (config::REPLENISH * args.dt) / 4.;
        while chance > 0. {
            if self.rng.gen_bool(chance.min(1.)) {
                let w: usize = self.rng.gen_range(0, config::FOOD_WIDTH);
                let h: usize = self.rng.gen_range(0, config::FOOD_HEIGHT);
                // expected: 4, chances is divided by 4
                // trying to get a less uniform food distribution
                let f: f64 = self.rng.gen_range(3., 5.);
                *self.foodgrid[w][h].get_mut() += f;
            }
            chance -= 1.;
        }

        // move blips
        let iter = self.blips.inner_mut().par_iter_mut();
        //let iter = self.blips.inner_mut().iter_mut();
        iter.flatten().for_each(|blip| blip.motion(args.dt));

        // update tree
        // todo: instead of re-building on each iteration i should update it
        // the blips have stable indices specifically for this usecase
        let tree = self
            .blips
            .iter_indexed()
            .inspect(|(_, b)| {
                assert!(!b.status.pos[0].is_nan());
                assert!(!b.status.pos[1].is_nan())
            })
            .map(|(index, b)| (b.status.pos, index))
            .collect();
        self.tree = SpatVec::new_from(tree);
    }
    pub fn report(&self) {
        use crate::select::Selection;
        let num = self.blips.len();
        if num == 0 {
            println!("no blips at all");
        } else {
            // todo: express this nicer once .collect into arrays is available
            let age = self
                .blips
                .get(
                    Selection::Age
                        .select(self.blips.iter_indexed(), self.tree(), &[0., 0.])
                        .unwrap(),
                )
                .unwrap()
                .status
                .age;
            let generation = self
                .blips
                .get(
                    Selection::Generation
                        .select(self.blips.iter_indexed(), self.tree(), &[0., 0.])
                        .unwrap(),
                )
                .unwrap()
                .status
                .generation;
            let spawns = self
                .blips
                .get(
                    Selection::Spawns
                        .select(self.blips.iter_indexed(), self.tree(), &[0., 0.])
                        .unwrap(),
                )
                .unwrap()
                .status
                .children;

            let food: f64 = self
                .foodgrid
                .iter()
                .flat_map(|a| a.iter())
                .map(|c| c.load(atomic::Ordering::Relaxed))
                .sum();

            let avg_food = food / ((config::FOOD_HEIGHT * config::FOOD_WIDTH) as f64);

            println!("number of blips    : {}", num);
            println!("oldest             : {}", age);
            println!("highest generation : {}", generation);
            println!("most reproduction  : {}", spawns);
            println!("total food         : {}", food);
            println!("average food       : {}", avg_food);
        }
    }
}
impl<B> PartialEq for App<B>
where
    B: Brain + Send + Copy + Sync + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        let base = self.blips == other.blips && self.tree == other.tree && self.time == other.time;
        if !base {
            return base;
        };
        self.foodgrid
            .iter()
            .zip(&other.foodgrid)
            .flat_map(|(s, o)| s.iter().zip(o.iter()))
            .all(|(s, o)| {
                let s = s.load(atomic::Ordering::Relaxed);
                let o = o.load(atomic::Ordering::Relaxed);
                s == o
            })
    }
}

#[test]
// this currently fails because floats are not commutative (a+b) + c != (a+c) + b
fn determinism() {
    use crate::brains::SimpleBrain;
    let mut app1 = App::<SimpleBrain>::new(1234);
    let mut app2 = App::<SimpleBrain>::new(1234);

    for _ in 0..1_000_000 {
        app1.update(&UpdateArgs { dt: 0.02 });
        app2.update(&UpdateArgs { dt: 0.02 });
        assert_eq!(app1, app2);
    }
}

#[test]
#[should_panic]
fn float_commu() {
    let mut rng = rand::thread_rng();
    let tests = 1_000;

    for _i in 0..tests {
        let a: f64 = rng.gen_range(-10., 10.);
        for _j in 0..tests {
            let b: f64 = rng.gen_range(-10., 10.);
            for _k in 0..tests {
                let c: f64 = rng.gen_range(-10., 10.);
                assert_eq!(a + b + c, a + c + b);
                assert_eq!(a - b - c, a - c - b);
                assert_eq!(a * b * c, a * c * b);
                assert_eq!(a / b / c, a / c / b);
            }
        }
    }
}

#[test]
#[should_panic]
// even storing as f32 and calculating in f64 does not help
fn float_commu_trunc() {
    let mut rng = rand::thread_rng();
    let tests = 1_000;

    for _i in 0..tests {
        let a: f32 = rng.gen_range(-10., 10.);
        for _j in 0..tests {
            let b: f32 = rng.gen_range(-10., 10.);
            for _k in 0..tests {
                let c: f32 = rng.gen_range(-10., 10.);
                let a = a as f64;
                let b = b as f64;
                let c = c as f64;
                assert_eq!(
                    ((a + b) as f32 as f64 + c) as f32,
                    ((a + c) as f32 as f64 + b) as f32
                );
                assert_eq!(
                    ((a - b) as f32 as f64 - c) as f32,
                    ((a - c) as f32 as f64 - b) as f32
                );
                assert_eq!(
                    ((a * b) as f32 as f64 * c) as f32,
                    ((a * c) as f32 as f64 * b) as f32
                );
                assert_eq!(
                    ((a / b) as f32 as f64 / c) as f32,
                    ((a / c) as f32 as f64 / b) as f32
                );
            }
        }
    }
}
