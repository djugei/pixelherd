use crate::config;

use crate::blip::{Blip, Genes, Status};
use crate::brains::Brain;
use crate::select::Selection;
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

pub fn foodenv(center: [usize; 2], size: isize) -> impl Iterator<Item = [usize; 2]> {
    (-size..=size)
        .flat_map(move |y| (-size..size).zip(std::iter::repeat(y)))
        .map(move |(x, y)| [center[0] as isize + x, center[1] as isize + y])
        .map(wrap_food)
}

pub fn wrap_food(coor: [isize; 2]) -> [usize; 2] {
    [
        wrap(coor[0], config::FOOD_WIDTH),
        wrap(coor[1], config::FOOD_HEIGHT),
    ]
}
// this only does one wrap, so v coordinate needs to not be too negative
pub fn wrap(v: isize, w: usize) -> usize {
    let v = if v < 0 { v + (w as isize) } else { v } as usize;
    v % w
}

#[derive(Debug)]
pub struct App<B: Brain + Send + Clone + Sync> {
    genes: Vec<Genes<B>>,
    status: Vec<Status>,
    vegtables: FoodGrid,
    // todo: consider giving meat a higher possible range (high risk/high reward)
    meat: FoodGrid,
    tree: Tree,
    time: f64,
    rng: DetRng,
    last_report: f64,
    report_file: Option<std::fs::File>,
}

impl<B: Brain + Send + Sync> App<B> {
    pub fn vegtables(&self) -> &FoodGrid {
        &self.vegtables
    }
    pub fn meat(&self) -> &FoodGrid {
        &self.meat
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
    pub fn new(seed: u64, report_path: Option<&str>) -> Self {
        let rng = DetRng::seed_from_u64(seed);
        let vegtables: OldFoodGrid =
            [[TenRat::from_int(0); config::FOOD_HEIGHT]; config::FOOD_WIDTH];
        let meat: OldFoodGrid = [[TenRat::from_int(0); config::FOOD_HEIGHT]; config::FOOD_WIDTH];
        //safety: this is absolutely not safe as i am relying on the internal memory layout of a third
        // party library that is almost guaranteed to not match on 32 bit platforms.
        //
        // however i see no other way to initialize this array
        // try_from is only implemented for array up to size 32 because fucking rust has no const
        // generics
        // atomics are not copy, so the [0.;times] constructor does not work
        // this is an actual value, not a reference so i need to actually change the value instead of
        // "as-casting" the pointer
        let vegtables = unsafe { std::mem::transmute(vegtables) };
        let meat = unsafe { std::mem::transmute(meat) };

        let report_file = report_path.map(std::fs::File::create).map(Result::unwrap);
        // Create a new simulation and run it.
        let mut app = App {
            status: Vec::with_capacity(config::INITIAL_CELLS),
            genes: Vec::with_capacity(config::INITIAL_CELLS),
            tree: SpatVec::new_from(Vec::with_capacity(config::INITIAL_CELLS)),
            vegtables,
            meat,
            time: 0.,
            last_report: 0.,
            rng,
            report_file,
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
                    *app.vegtables[w][h].get_mut() = app.rng.gen_range(0., 10.).into();
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

        let mut oldveg: OldFoodGrid = [[TenRat::from(0); config::FOOD_HEIGHT]; config::FOOD_WIDTH];
        // maybe split this into two parts, a pure memcopy and the modification/clamp
        for (w, r) in oldveg.iter_mut().zip(self.vegtables.iter_mut()) {
            for (w, r) in w.iter_mut().zip(r.iter_mut()) {
                // clamp to valid range on each iteration
                // datatype is valid from -16 to 16
                // but 0-10 is the only sensible value range for application domain
                // this is important for determinism as saturations/wraparounds break determinism
                // (well except we only do subtraction, so saturating would be fine)
                *w = (*r.get_mut()).clamp((-1).into(), 12.into());
            }
        }

        let mut oldmeat: OldFoodGrid = [[TenRat::from(0); config::FOOD_HEIGHT]; config::FOOD_WIDTH];
        // maybe split this into two parts, a pure memcopy and the modification/clamp
        for (w, r) in oldmeat.iter_mut().zip(self.meat.iter_mut()) {
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
                &oldveg,
                &self.vegtables,
                &oldmeat,
                &self.meat,
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
        while chance > 0. {
            if self.rng.gen_bool(chance.min(1.)) {
                let w: usize = self.rng.gen_range(0, config::FOOD_WIDTH);
                let h: usize = self.rng.gen_range(0, config::FOOD_HEIGHT);
                let grid = self.vegtables[w][h].get_mut();
                // expected: 4, chances is divided by 4
                // trying to get a less uniform food distribution
                let f: TenRat = self.rng.gen_range(3., 5.).into();
                *grid = grid.saturating_add(f);
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

        // just to make sure ppl with old rust versions can't run this, fuck debian in particular
        if self.time - self.last_report > std::f64::consts::TAU * 100. {
            self.write_report();
            self.last_report = self.time;
            // write to stdout more rarely
            if self.time as u64 % ((std::f64::consts::TAU * 1000.) as u64) == 0 {
                self.report()
            }
        }
    }
}
#[derive(Debug, PartialEq)]
pub struct Report {
    time: f64,
    num: usize,
    age: f64,
    generation: usize,
    spawns: usize,
    lineage: f64,
    veg: f64,
    avg_veg: f64,
    meat: f64,
    avg_meat: f64,
}
impl<B: Brain + Send + Sync> App<B> {
    // maybe return Iterater<Item = (Selection, Status)> instead
    fn gen_report(&self) -> Option<Report> {
        if self.status.len() == 0 {
            return Option::None;
        }
        use Selection::*;
        let selections = [Age, Generation, Spawns, Lineage];
        let mut s = selections
            .iter()
            .map(|s| {
                s.select(self.status.iter().enumerate(), self.tree(), &[0., 0.])
                    .unwrap()
            })
            .map(|pos| self.status.get(pos).unwrap());

        let veg: f64 = self
            .vegtables
            .iter()
            .flat_map(|a| a.iter())
            .map(|c| c.load(atomic::Ordering::Relaxed).to_f64())
            .sum();
        let meat: f64 = self
            .meat
            .iter()
            .flat_map(|a| a.iter())
            .map(|c| c.load(atomic::Ordering::Relaxed).to_f64())
            .sum();
        let fields = (config::FOOD_HEIGHT * config::FOOD_WIDTH) as f64;
        Report {
            time: self.time,
            num: self.status.len(),
            age: s.next().unwrap().age,
            generation: s.next().unwrap().generation,
            spawns: s.next().unwrap().children,
            lineage: s.next().unwrap().lineage,
            meat,
            veg,
            avg_veg: veg / fields,
            avg_meat: meat / fields,
        }
        .into()
    }

    pub fn report(&self) {
        if let Some(r) = self.gen_report() {
            println!("report for         : {}", r.time);
            println!("number of blips    : {}", r.num);
            println!("oldest             : {}", r.age);
            println!("highest generation : {}", r.generation);
            println!("most reproduction  : {}", r.spawns);
            println!("longest lineage    : {}", r.lineage);
            println!("total veg          : {}", r.veg);
            println!("average veg        : {}", r.avg_veg);
            println!("total meat         : {}", r.meat);
            println!("average meat       : {}", r.avg_meat);
            println!();
        } else {
            println!("no blips at all");
        }
    }

    // takes &mut for the file write, kinda pointless since files are global state anyway
    pub fn write_report(&mut self) {
        if self.report_file.is_none() {
            return;
        }
        let r = if let Some(r) = self.gen_report() {
            r
        } else {
            return;
        };
        // can only do the match here cause borrow checker is not smart enough to notice that
        // report_file is never accessed from the gen_report function.
        if let Some(file) = self.report_file.as_mut() {
            let reportline = format!(
                "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n",
                r.time,
                r.num,
                r.age,
                r.generation,
                r.spawns,
                r.lineage,
                r.veg,
                r.avg_veg,
                r.meat,
                r.avg_meat
            );
            std::io::Write::write_all(file, reportline.as_bytes()).unwrap();
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
        self.vegtables
            .iter()
            .zip(&other.vegtables)
            .flat_map(|(s, o)| s.iter().zip(o.iter()))
            .all(|(s, o)| {
                let s = s.load(atomic::Ordering::Relaxed);
                let o = o.load(atomic::Ordering::Relaxed);
                s == o
            })
            && self
                .meat
                .iter()
                .zip(&other.meat)
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
    let mut app1 = App::<SimpleBrain>::new(1234, None);
    let mut app2 = App::<SimpleBrain>::new(1234, None);

    for i in 0..20_000 {
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
