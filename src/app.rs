use crate::config;

use crate::blip::{Blip, Genes, Status};
use crate::brains::Brain;
use anti_r::vec::SpatVec;
use anti_r::SpatSlice;
use atomic::Atomic;
use fix_rat::TenRat;
use piston::input::UpdateArgs;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng as DetRng;
use rayon::prelude::IndexedParallelIterator;
use rayon::prelude::IntoParallelRefMutIterator;
use rayon::prelude::ParallelIterator;

#[test]
fn tenrat_atomic() {
    assert!(Atomic::<TenRat>::is_lock_free());
}
pub type Tree = SpatVec<[f64; 2], usize>;
pub type TreeRef<'a> = SpatSlice<'a, [f64; 2], usize>;

// using TenRat instead of f64 here to get associativity, which is important cause of multithreading
// todo: consider creating a real type for the foodgrid
// each step foodgrid is copied into a read only and a (synchronized) write-only part
pub type FoodGrid = [[Atomic<TenRat>; config::FOOD_HEIGHT]; config::FOOD_WIDTH];
// safety: always change this in sync
pub type OldFoodGrid = [[TenRat; config::FOOD_HEIGHT]; config::FOOD_WIDTH];

#[derive(Debug)]
pub struct App<B: Brain + Send + Clone + Sync> {
    genes: Vec<Genes<B>>,
    status: Vec<Status>,
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
    pub fn tree(&self) -> TreeRef<'_> {
        (&self.tree).into()
    }
    pub fn blips_ro(&self) -> impl Iterator<Item = (&Status, &Genes<B>)> {
        self.status.iter().zip(&self.genes)
    }
    pub fn statuses(&self) -> &[Status] {
        &self.status
    }
    pub fn new(seed: u64) -> Self {
        let rng = DetRng::seed_from_u64(seed);
        let foodgrid: OldFoodGrid =
            [[TenRat::from_int(0); config::FOOD_HEIGHT]; config::FOOD_WIDTH];
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
            status: Vec::with_capacity(config::INITIAL_CELLS),
            genes: Vec::with_capacity(config::INITIAL_CELLS),
            tree: SpatVec::new_from(Vec::with_capacity(config::INITIAL_CELLS)),
            foodgrid,
            time: 0.,
            rng,
        };

        for _ in 0..config::INITIAL_CELLS {
            let (s, g) = Blip::new(&mut app.rng);
            app.status.push(s);
            app.genes.push(g);
        }

        for w in 0..config::FOOD_WIDTH {
            for h in 0..config::FOOD_HEIGHT {
                //fixme: this should be an exponential distribution instead
                if app.rng.gen_range(0, 3) == 1 {
                    *app.foodgrid[w][h].get_mut() = app.rng.gen_range(0., 10.).into();
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
        let mut new = self.status.clone();

        let spawns = std::sync::Mutex::new(Vec::new());

        let iter = new
            .par_iter_mut()
            .zip(&self.status)
            .zip(&self.genes)
            .enumerate()
            .map(|(index, ((new, old), genes))| (index, old, Blip::from_components(new, genes)));

        let mut oldgrid: OldFoodGrid = [[TenRat::from(0); config::FOOD_HEIGHT]; config::FOOD_WIDTH];
        for (w, r) in oldgrid.iter_mut().zip(self.foodgrid.iter_mut()) {
            for (w, r) in w.iter_mut().zip(r.iter_mut()) {
                // clamp to valid range on each iteration
                // datatype is valid from -16 to 16
                // but 0-10 is the only sensible value range for application domain
                // this is important for determinism as saturations/wraparounds break determinism
                // (well except we only do subtraction, so saturating would be fine)
                *w = (*r.get_mut()).clamp((-1).into(), 12.into());
            }
        }

        // new is write only. if you need data from the last iteration
        // get it from old only.
        iter.for_each(|(index, old, mut new)| {
            let seed = self.time.to_bits();
            let seed = seed ^ (index as u64);
            // todo: better seeding, can seed from root rng, store rng with blip
            let mut rng = DetRng::seed_from_u64(seed);

            let spawn = new.update(
                &mut rng,
                old,
                &self.status,
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

        for (_i, (s, g)) in spawns {
            // oh fuck the indices gotta really make sure i don't fuck this up
            new.push(s);
            self.genes.push(g);
        }

        let deaths = new
            .iter()
            .enumerate()
            .filter(|(_i, s)| s.hp <= 0.)
            .map(|(i, _)| i)
            .rev()
            .collect::<Vec<_>>();

        for i in deaths {
            // todo: drop some meat on death
            new.remove(i);
            self.genes.remove(i);
        }

        if new.len() < config::REPLACEMENT {
            println!("force-spawned");
            let (s, g) = Blip::new(&mut self.rng);
            new.push(s);
            self.genes.push(g);
        }

        self.status = new;

        // chance could be > 1 if dt or replenish are big enough
        let mut chance = (config::REPLENISH * args.dt) / 4.;
        // fixme: this overruns the rational (generally when food is dropped 2x on the same field)
        while chance > 0. {
            if self.rng.gen_bool(chance.min(1.)) {
                let w: usize = self.rng.gen_range(0, config::FOOD_WIDTH);
                let h: usize = self.rng.gen_range(0, config::FOOD_HEIGHT);
                let grid = self.foodgrid[w][h].get_mut();
                // expected: 4, chances is divided by 4
                // trying to get a less uniform food distribution
                let f: TenRat = self.rng.gen_range(3., 5.).into();
                *grid = *grid + f;
            }
            chance -= 1.;
        }

        // move blips
        let iter = self
            .status
            .par_iter_mut()
            .zip(&self.genes)
            .map(|(s, g)| Blip::from_components(s, g));
        iter.for_each(|mut blip| blip.motion(args.dt));

        // update tree
        // todo: instead of re-building on each iteration i should update it
        // the blips have stable indices specifically for this usecase
        let tree = self
            .status
            .iter()
            .enumerate()
            .inspect(|(_, s)| {
                assert!(!s.pos[0].is_nan());
                assert!(!s.pos[1].is_nan())
            })
            .map(|(index, s)| (s.pos, index))
            .collect();
        self.tree = SpatVec::new_from(tree);
    }
    pub fn report(&self) {
        use crate::select::Selection;
        let num = self.genes.len();
        if num == 0 {
            println!("no blips at all");
        } else {
            // todo: express this nicer once .collect into arrays is available
            let age = self
                .status
                .get(
                    Selection::Age
                        .select(self.status.iter().enumerate(), self.tree(), &[0., 0.])
                        .unwrap(),
                )
                .unwrap()
                .age;
            let generation = self
                .status
                .get(
                    Selection::Generation
                        .select(self.status.iter().enumerate(), self.tree(), &[0., 0.])
                        .unwrap(),
                )
                .unwrap()
                .generation;
            let spawns = self
                .status
                .get(
                    Selection::Spawns
                        .select(self.status.iter().enumerate(), self.tree(), &[0., 0.])
                        .unwrap(),
                )
                .unwrap()
                .children;

            let food: f64 = self
                .foodgrid
                .iter()
                .flat_map(|a| a.iter())
                .map(|c| c.load(atomic::Ordering::Relaxed).to_f64())
                .sum();

            let fields = (config::FOOD_HEIGHT * config::FOOD_WIDTH) as f64;

            let avg_food = food / fields;

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
        let base = self.status == other.status
            && self.genes == other.genes
            && self.tree == other.tree
            && self.time == other.time;
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
fn determinism() {
    use crate::brains::SimpleBrain;
    let mut app1 = App::<SimpleBrain>::new(1234);
    let mut app2 = App::<SimpleBrain>::new(1234);

    for i in 0..100_000 {
        if i % 100 == 0 {
            println!("determinism iteration {}", i);
        }
        app1.update(&UpdateArgs { dt: 0.02 });
        app2.update(&UpdateArgs { dt: 0.02 });
        if app1 != app2 {
            use std::fs::File;
            use std::io::Write;
            let mut f1 = File::create("dump_app1").unwrap();
            let mut f2 = File::create("dump_app2").unwrap();

            let s1 = format!("{:#?}", app1);
            let s2 = format!("{:#?}", app2);

            f1.write_all(s1.as_bytes()).unwrap();
            f2.write_all(s2.as_bytes()).unwrap();
        }
        assert_eq!(app1, app2);
    }
}

#[test]
#[should_panic]
fn float_assoc() {
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
fn float_assoc_trunc() {
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
