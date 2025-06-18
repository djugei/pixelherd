use crate::config;

use crate::blip::{Blip, Genes, Status};
use crate::brains::Brain;
use crate::select::Selection;
use anti_r::SpatSlice;
use anti_r::SpatVec;
use atomic::Atomic;
use fix_rat::{HundRat, TenRat};
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg as DetRng;
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
type FoodGrid<T> = [[Atomic<T>; config::FOOD_HEIGHT]; config::FOOD_WIDTH];
// safety: always change this in sync
type OldFoodGrid<T> = [[T; config::FOOD_HEIGHT]; config::FOOD_WIDTH];

pub type VegGrid = FoodGrid<TenRat>;
pub type OldVegGrid = OldFoodGrid<TenRat>;

pub type MeatGrid = FoodGrid<HundRat>;
pub type OldMeatGrid = OldFoodGrid<HundRat>;

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

use bigmatrix::BigMatrix;
use serde::{de::DeserializeOwned, Serialize};
use serde_derive as sd;

#[derive(sd::Serialize, sd::Deserialize, Clone, Debug)]
pub struct SerializeApp<B: Brain> {
    genes: Vec<Genes<B>>,
    status: Vec<Status>,
    #[serde(with = "BigMatrix")]
    vegtables: OldVegGrid,
    #[serde(with = "BigMatrix")]
    meat: OldMeatGrid,
    time: u64,
    rng: DetRng,
    last_report: u64,
}

///SAFETY: only call this if T and Atomic<T> have the same in-memory representation
unsafe fn food_s_to_a<T: Sized>(g: &OldFoodGrid<T>) -> FoodGrid<T> {
    unsafe {
        std::mem::transmute_copy(g)
    }
}

///SAFETY: only call this if T and Atomic<T> have the same in-memory representation
///the &mut is to force the atomic to not be in other threads
unsafe fn food_a_to_s<T: Default>(g: &mut FoodGrid<T>) -> OldFoodGrid<T> {
    unsafe {
        std::mem::transmute_copy(g)
    }
}

impl<B: Brain + Send + Sync + Serialize + DeserializeOwned> SerializeApp<B> {
    fn unpack(self, report_file: Option<std::fs::File>) -> App<B> {
        // safety: no, i can not guarantee that T and Atomic<T> have the same representation,
        // especially on 32-bit platforms. arrays are so annoying to deal with that i am ignoring
        // that though.
        let vegtables = unsafe { food_s_to_a(&self.vegtables) };
        let meat = unsafe { food_s_to_a(&self.meat) };
        let mut a = App {
            genes: self.genes,
            status: self.status,
            time: self.time,
            rng: self.rng,
            last_report: self.last_report,
            vegtables,
            meat,
            tree: SpatVec::new_from(Vec::new()),
            report_file,
        };
        a.update_tree();
        a
    }
    fn pack(other: &mut App<B>) -> Self {
        // safety: no, i can not guarantee that T and Atomic<T> have the same representation,
        // especially on 32-bit platforms. arrays are so annoying to deal with that i am ignoring
        // that though.
        let vegtables = unsafe { food_a_to_s(&mut other.vegtables) };
        let meat = unsafe { food_a_to_s(&mut other.meat) };
        Self {
            genes: other.genes.clone(),
            status: other.status.clone(),
            time: other.time,
            rng: other.rng.clone(),
            last_report: other.last_report,
            vegtables,
            meat,
        }
    }
}

#[derive(Debug)]
pub struct App<B: Brain + Send + Sync> {
    genes: Vec<Genes<B>>,
    status: Vec<Status>,
    vegtables: VegGrid,
    meat: MeatGrid,
    tree: Tree,
    time: u64,
    rng: DetRng,
    last_report: u64,
    report_file: Option<std::fs::File>,
}
impl<B: Brain + Send + Sync + Serialize + DeserializeOwned> App<B> {
    pub fn new_from<R: std::io::Read>(
        mut r: R,
        report_path: Option<&str>,
    ) -> Result<Self, bincode::error::DecodeError> {
        let report_file = report_path.map(std::fs::File::create).map(Result::unwrap);
        let sa: SerializeApp<_> =
            bincode::serde::decode_from_std_read(&mut r, bincode::config::standard())?;
        let s = sa.unpack(report_file);
        Ok(s)
    }
    pub fn write_into<W: std::io::Write>(
        &mut self,
        mut w: W,
    ) -> Result<usize, bincode::error::EncodeError> {
        bincode::serde::encode_into_std_write(
            &SerializeApp::pack(self),
            &mut w,
            bincode::config::standard(),
        )
    }
}

impl<B: Brain + Send + Sync> App<B> {
    pub fn vegtables(&self) -> &VegGrid {
        &self.vegtables
    }
    pub fn meat(&self) -> &MeatGrid {
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
        let vegtables: OldVegGrid =
            [[TenRat::from_int(0); config::FOOD_HEIGHT]; config::FOOD_WIDTH];
        let meat: OldMeatGrid = [[HundRat::from_int(0); config::FOOD_HEIGHT]; config::FOOD_WIDTH];
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
            time: 0,
            last_report: 0,
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
                if app.rng.random_range(0..3) == 1 {
                    *app.vegtables[w][h].get_mut() = app.rng.random_range(0.0..10.).into();
                }
            }
        }
        app
    }

    pub fn update(&mut self) {
        self.time += 1;
        // update the inputs
        // todo: don't clone here, keep two buffers and swap instead
        let mut new = self.status.clone();

        let spawns = std::sync::Mutex::new(Vec::new());

        let iter = new
            .par_iter_mut()
            .zip(&self.genes)
            .enumerate()
            .map(|(index, (new, genes))| (index, Blip::from_components(new, genes)));

        // todo: this is the most calculation intensive sequential part, maybe utilising some
        // unsafe/maybeuninit would be worth it.
        let mut oldveg: Box<OldVegGrid> =
            Box::new([[TenRat::from(0); config::FOOD_HEIGHT]; config::FOOD_WIDTH]);
        let mut oldmeat: Box<OldMeatGrid> =
            Box::new([[HundRat::from(0); config::FOOD_HEIGHT]; config::FOOD_WIDTH]);

        let vegs = self.vegtables.par_iter_mut();
        let meats = self.meat.par_iter_mut();
        rayon::join(
            || {
                oldveg.par_iter_mut().zip(vegs).for_each(|(w, r)| {
                    for (w, r) in w.iter_mut().zip(r.iter_mut()) {
                        // clamp to valid range on each iteration
                        // datatype is valid from -16 to 16
                        // but 0-10 is the only sensible value range for application domain
                        // this is important for determinism as saturations break determinism
                        // (well except we only do subtraction, so saturating would be fine)
                        *w = (*r.get_mut()).clamp((-1).into(), 12.into());
                    }
                })
            },
            || {
                oldmeat.par_iter_mut().zip(meats).for_each(|(w, r)| {
                    for (w, r) in w.iter_mut().zip(r.iter_mut()) {
                        // clamp to valid range on each iteration
                        // datatype is valid from -128 to 128
                        // this is important for determinism as saturations break determinism
                        // meat actually gets subtracted (eaten) and added to (deaths)
                        *w = (*r.get_mut()).clamp((-1).into(), 70.into());
                    }
                })
            },
        );

        // new is write only. if you need data from the last iteration
        // get it from old only.
        iter.for_each(|(index, mut new)| {
            let seed = self.time.to_le();
            let seed = seed ^ (index as u64);
            // todo: better seeding, can seed from root rng, store rng with blip
            let mut rng = DetRng::seed_from_u64(seed);

            let spawn = new.update(
                &mut rng,
                &self.status,
                &self.genes,
                self.tree(),
                &oldveg,
                &self.vegtables,
                &oldmeat,
                &self.meat,
                self.time,
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
            // gotta make sure the blips (status) and genes indices always match up
            new.push(s);
            self.genes.push(g);
        }

        // can't just do retain as I need to drop from both new and genes.
        // notice the rev() call though so all is fine
        let deaths = new
            .iter()
            .enumerate()
            .filter(|(_i, s)| s.hp <= 0.)
            .map(|(i, _)| i)
            .rev()
            .collect::<Vec<_>>();

        for i in deaths {
            new.remove(i);
            self.genes.remove(i);
        }

        // make sure there is at least some vegetables and meat eaters each
        let (min, max) = self
            .genes
            .iter()
            .map(|g| g.vore)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |old, new| {
                (old.0.min(new), old.1.max(new))
            });
        if min > 0.5 {
            let (s, mut g) = Blip::new(&mut self.rng);
            g.vore = (g.vore - 0.1).max(0.);
            new.push(s);
            self.genes.push(g);
        }
        if max < 0.5 {
            let (s, mut g) = Blip::new(&mut self.rng);
            g.vore = (g.vore + 0.1).min(1.);
            new.push(s);
            self.genes.push(g);
        }

        if new.len() < config::REPLACEMENT {
            //println!("force-spawned");
            let (s, g) = Blip::new(&mut self.rng);
            new.push(s);
            self.genes.push(g);
        }

        self.status = new;

        // chance could be > 1 if dt or replenish are big enough
        let mut chance = (config::REPLENISH * config::STEP_SIZE) / 4.;
        while chance > 0. {
            if self.rng.random_bool(chance.min(1.)) {
                let w: usize = self.rng.random_range(0..config::FOOD_WIDTH);
                let h: usize = self.rng.random_range(0..config::FOOD_HEIGHT);
                let grid = self.vegtables[w][h].get_mut();
                // expected: 4, chances is divided by 4
                // trying to get a less uniform food distribution
                let f: TenRat = self.rng.random_range(3.0..5.).into();
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
        iter.for_each(|mut blip| blip.motion());

        // update tree
        self.update_tree();

        // just to make sure ppl with old rust versions can't run this, fuck debian in particular
        if self.time - self.last_report > (std::f64::consts::TAU * 100.) as u64 {
            self.write_report();
            self.last_report = self.time;
            // write to stdout more rarely
            if self.time as u64 % ((std::f64::consts::TAU * 1000.) as u64) == 0 {
                self.report()
            }
        }
    }
    fn update_tree(&mut self) {
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
        // todo: blips close to any edge should be inserted twice,
        // currently blips can't look/collide across edges.
        // different option would be to fix this in the distance calculations.
        self.tree = SpatVec::new_from(tree);
    }
}
#[derive(Debug, PartialEq)]
pub struct Report {
    time: u64,
    num: usize,
    age: u64,
    generation: usize,
    spawns: usize,
    lineage: u64,
    veg: f64,
    avg_veg: f64,
    meat: f64,
    avg_meat: f64,
}
impl<B: Brain + Send + Sync> App<B> {
    // maybe return Iterator<Item = (Selection, Status)> instead
    fn gen_report(&self) -> Option<Report> {
        if self.status.is_empty() {
            return Option::None;
        }
        use Selection::*;
        let selections = [Age, Generation, Spawns, Lineage];
        let mut s = selections
            .iter()
            .map(|s| {
                s.select(self.blips_ro().enumerate(), self.tree(), &[0., 0.])
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
    std::thread::Builder::new()
        // i don't quite understand why the stacks have to be so big...
        // this is much more than 2 x App
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            let mut app1 = App::<SimpleBrain>::new(1234, None);
            let mut app2 = App::<SimpleBrain>::new(1234, None);

            for i in 0..20_000 {
                if i % 100 == 0 {
                    println!("determinism iteration {}", i);
                }
                app1.update();
                app2.update();
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
        })
        .unwrap()
        .join()
        .unwrap()
}

#[test]
#[should_panic]
fn float_assoc() {
    let mut rng = rand::rng();
    let tests = 1_000;

    for _i in 0..tests {
        let a: f64 = rng.random_range(-10.0..10.);
        for _j in 0..tests {
            let b: f64 = rng.random_range(-10.0..10.);
            for _k in 0..tests {
                let c: f64 = rng.random_range(-10.0..10.);
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
    let mut rng = rand::rng();
    let tests = 1_000;

    for _i in 0..tests {
        let a: f32 = rng.random_range(-10.0..10.);
        for _j in 0..tests {
            let b: f32 = rng.random_range(-10.0..10.);
            for _k in 0..tests {
                let c: f32 = rng.random_range(-10.0..10.);
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

#[test]
fn matrix_ser_de() {
    #[derive(Debug, Serialize, sd::Deserialize, PartialEq, Eq, Copy, Clone)]
    struct T {
        #[serde(with = "BigMatrix")]
        e: [[usize; 33]; 44],
    }
    let mut t: T = T {
        e: [[Default::default(); 33]; 44],
    };

    use rand::distr::Distribution;
    use std::convert::TryFrom;
    let mut rng = rand::rng();
    let range = rand::distr::Uniform::try_from(0..usize::MAX).unwrap();
    for l in t.e.iter_mut() {
        for e in l.iter_mut() {
            *e = range.sample(&mut rng);
        }
    }
    let t = t;
    use bincode;
    let ser: Vec<u8> = bincode::serde::encode_to_vec(&t, bincode::config::standard()).unwrap();
    let (de, _) = bincode::serde::decode_from_slice(&ser, bincode::config::standard()).unwrap();
    assert_eq!(t, de);
}

#[test]
fn ser_de_determinism() {
    use crate::brains::SimpleBrain;
    let stacksize = core::mem::size_of::<App<SimpleBrain>>();
    let stacksize = stacksize * 5;
    std::thread::Builder::new()
        // in theory this should be enough,
        // 2x app, one serialized app, one deserialized (should not exist in parallel)
        // and a bit of extra space.
        .stack_size(stacksize)
        // however even 8 mb is not enough (overflows on first serialization)
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let mut app1 = App::<SimpleBrain>::new(1234, None);
            let mut app2 = App::<SimpleBrain>::new(1234, None);

            for i in 0..20_000 {
                if i % 100 == 0 {
                    println!("serde determinism iteration {}", i);
                }
                app1.update();
                app2.update();
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
                if i % 1000 == 0 {
                    let a2s = bincode::serde::encode_to_vec(
                        &SerializeApp::pack(&mut app2),
                        bincode::config::standard(),
                    )
                    .unwrap();
                    let (a2d, _s): (SerializeApp<_>, usize) =
                        bincode::serde::decode_from_slice(&a2s, bincode::config::standard())
                            .unwrap();
                    app2 = a2d.unpack(None);
                }
            }
        })
        .unwrap()
        .join()
        .unwrap()
}
