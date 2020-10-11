use crate::config;

use crate::blip::Blip;
use crate::brains::Brain;
use crate::stablevec::StableVec;
use crate::vecmath;
use crate::vecmath::Vector;
use atomic::Atomic;
use piston::input::UpdateArgs;
use rand::Rng;
use rayon::prelude::IndexedParallelIterator;
use rayon::prelude::IntoParallelRefMutIterator;
use rayon::prelude::ParallelIterator;
use rstar::primitives::PointWithData;
use rstar::RTree;

pub type BlipLoc = PointWithData<usize, [f64; 2]>;
// each step foodgrid is copied into a read only and a (synchronized) write-only part
pub type FoodGrid = [[Atomic<f64>; config::FOOD_HEIGHT]; config::FOOD_WIDTH];
// safety: always change this in sync
pub type OldFoodGrid = [[f64; config::FOOD_HEIGHT]; config::FOOD_WIDTH];

pub struct App<B: Brain + Send + Copy + Sync> {
    blips: StableVec<Blip<B>>,
    foodgrid: FoodGrid,
    // todo: replace with a simple sorted list
    tree: RTree<BlipLoc>,
    time: f64,
}

impl<B: Brain + Send + Copy + Sync> App<B> {
    pub fn foodgrid(&self) -> &FoodGrid {
        &self.foodgrid
    }
    pub fn tree(&self) -> &RTree<BlipLoc> {
        &self.tree
    }
    pub fn blips(&self) -> &StableVec<Blip<B>> {
        &self.blips
    }
    pub fn new<R: Rng>(mut rng: R) -> Self {
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
            tree: RTree::new(),
            foodgrid,
            time: 0.,
        };

        for _ in 0..config::INITIAL_CELLS {
            app.blips.push(Blip::new(&mut rng));
        }

        for w in 0..config::FOOD_WIDTH {
            for h in 0..config::FOOD_HEIGHT {
                //fixme: this should be an exponential distribution instead
                if rng.gen_range(0, 3) == 1 {
                    *app.foodgrid[w][h].get_mut() = rng.gen_range(0., 10.);
                }
            }
        }
        app
    }
    // fixme: make sure dt is used literally on every change
    pub fn update<R: Rng>(&mut self, args: &UpdateArgs, mut rng: R) {
        self.time += args.dt;
        // update the inputs
        // todo: don't clone here, keep two buffers and swap instead
        let mut new = self.blips.clone();

        let spawns = std::sync::Mutex::new(Vec::new());

        //perf: benchmarks if more cpu = more speed
        let iter = new.inner_mut().par_iter_mut().zip(self.blips.inner());
        //let iter = new.iter_mut().zip(&self.blips);

        let mut oldgrid: OldFoodGrid = [[0.; config::FOOD_HEIGHT]; config::FOOD_WIDTH];
        for (w, r) in oldgrid.iter_mut().zip(self.foodgrid.iter_mut()) {
            for (w, r) in w.iter_mut().zip(r.iter_mut()) {
                *w = *r.get_mut();
            }
        }

        // new is write only. if you need data from the last iteration
        // get it from old only.
        iter.flatten().for_each(|(new, old)| {
            // todo: figure out how to pass rng into other threads
            let mut rng = rand::thread_rng();
            let spawn = new.update(
                &mut rng,
                old,
                &self.blips,
                &self.tree,
                &oldgrid,
                &self.foodgrid,
                self.time,
                args.dt,
            );
            if let Some(spawn) = spawn {
                let mut guard = spawns.lock().unwrap();
                guard.push(spawn);
            }
        });

        let spawns = spawns.into_inner().unwrap();

        new.extend(spawns);

        // todo: drop some meat on death
        new.retain(|blip| blip.status.hp > 0.);

        if new.len() < config::REPLACEMENT {
            println!("force-spawned");
            new.push(Blip::new(&mut rng));
        }

        self.blips = new;

        // chance could be > 1 if dt or replenish are big enough
        let mut chance = (config::REPLENISH * args.dt) / 4.;
        while chance > 0. {
            if rng.gen_bool(chance.min(1.)) {
                let w: usize = rng.gen_range(0, config::FOOD_WIDTH);
                let h: usize = rng.gen_range(0, config::FOOD_HEIGHT);
                // expected: 4, chances is divided by 4
                // trying to get a less uniform food distribution
                let f: f64 = rng.gen_range(3., 5.);
                *self.foodgrid[w][h].get_mut() += f;
            }
            chance -= 1.;
        }

        // move blips
        let iter = self.blips.inner_mut().par_iter_mut();
        iter.flatten().for_each(|blip| blip.motion(args.dt));

        // update tree
        let tree = self
            .blips
            .iter_indexed()
            .inspect(|(_, b)| {
                assert!(!b.status.pos[0].is_nan());
                assert!(!b.status.pos[1].is_nan())
            })
            .map(|(p, b)| BlipLoc::new(p, b.status.pos))
            .collect();
        self.tree = RTree::bulk_load(tree);
    }
    pub fn report(&self) {
        use crate::select::Selection;
        let num = self.blips.len();
        if num == 0 {
            println!("no blips at all");
        } else {
            let age = self
                .blips
                .get(
                    Selection::Age
                        .select(self.blips.iter_indexed(), &self.tree, &[0., 0.])
                        .unwrap(),
                )
                .unwrap()
                .status
                .age;
            let generation = self
                .blips
                .get(
                    Selection::Generation
                        .select(self.blips.iter_indexed(), &self.tree, &[0., 0.])
                        .unwrap(),
                )
                .unwrap()
                .status
                .generation;
            let spawns = self
                .blips
                .get(
                    Selection::Spawns
                        .select(self.blips.iter_indexed(), &self.tree, &[0., 0.])
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

pub fn locate_in_radius(
    tree: &RTree<BlipLoc>,
    center: Vector,
    env: f64,
) -> impl Iterator<Item = (&BlipLoc, f64)> {
    let lu = vecmath::add(center, [-env, -env]);
    let rd = vecmath::add(center, [env, env]);

    let bb = rstar::AABB::from_corners(lu, rd);

    use rstar::PointDistance;
    tree.locate_in_envelope(&bb)
        .map(move |p| (p, p.position().distance_2(&center)))
        .filter(move |(_p, d)| *d <= env)
}
