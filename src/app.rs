use crate::config::*;

use crate::blip::Blip;
use crate::brains::BigBrain;
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
// the foodgrid is currently accessed in parallel. while there are no classical dataraces (to my
// knowledge) the behaviour is different from a linear execution:
// a "later" actor may have accessed and updated a field before an "earlier" one from a different
// thread got to execute.
// fixing that, forcing step size to a constant number, and "determinizing" the rng would make
// the execution entirely deterministic, which might be a desireable property.
// to do so the food grid would need to store for each field the highest id that accessed it this
// step.
// if a blip tries to access a food grid field thats marked with a higher id than itself it needs
// to undo that:
// when accessing a field each blip stores the previous id.
// when a blip detects a conflict, it follows the chain in order, re-calculating each blip,
// then inserting its own modification at the appropriate location, and afterwards running all
// blips with a higher id.
// this is obviously quite expensive, and might even have to be done multiple times.
// it _can_ be done in parallel though so if running it on the worker thread or on the main thread
// (which would limit the number of times such a re-execution needs to happen to one/field) depends
// on the expected contention and number of available cpus
//
// in the opposite instead of re-cacluating when noticing a concurrent access we could just
// re-subtract from the new value if determinism is not a concern
pub type FoodGrid = [[Atomic<f64>; FOOD_HEIGHT]; FOOD_WIDTH];

pub struct App {
    blips: Vec<Blip<BigBrain>>,
    foodgrid: FoodGrid,
    tree: RTree<BlipLoc>,
    time: f64,
}

impl App {
    pub fn foodgrid(&self) -> &FoodGrid {
        &self.foodgrid
    }
    pub fn tree(&self) -> &RTree<BlipLoc> {
        &self.tree
    }
    pub fn blips(&self) -> &[Blip<BigBrain>] {
        &self.blips
    }
    pub fn new<R: Rng>(mut rng: R) -> Self {
        let foodgrid = [[0.; FOOD_HEIGHT]; FOOD_WIDTH];
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
            blips: Vec::with_capacity(INITIAL_CELLS),
            tree: RTree::new(),
            foodgrid,
            time: 0.,
        };

        for _ in 0..INITIAL_CELLS {
            app.blips.push(Blip::new(&mut rng));
        }

        for w in 0..FOOD_WIDTH {
            for h in 0..FOOD_HEIGHT {
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
        if args.dt > 0.3 {
            println!("we are lagging, {} s/tick", args.dt);
        }
        self.time += args.dt;
        // update the inputs
        // todo: don't clone here, keep two buffers and swap instead
        let mut new = self.blips.clone();

        let spawns = std::sync::Mutex::new(Vec::new());

        //perf: benchmarks if more cpu = more speed
        let iter = new.par_iter_mut().zip(&self.blips);
        //let iter = new.iter_mut().zip(&self.blips);

        // new is write only. if you need data from the last iteration
        // get it from old only.
        iter.for_each(|(new, old)| {
            // todo: figure out how to pass rng into other threads
            let mut rng = rand::thread_rng();
            let spawn = new.update(
                &mut rng,
                old,
                &self.blips,
                &self.tree,
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
        let before = new.len();
        new.retain(|blip| blip.status.hp > 0.);
        let after = new.len();
        if after < before {
            println!("{} deaths", before - after);
        }

        if new.len() < REPLACEMENT {
            new.push(Blip::new(&mut rng));
        }

        self.blips = new;

        // add some food
        // fixme: if dt * factor > 1 this needs to run multiple times
        if rng.gen_bool(args.dt) {
            let w: usize = rng.gen_range(0, FOOD_WIDTH);
            let h: usize = rng.gen_range(0, FOOD_HEIGHT);
            let f: f64 = rng.gen_range(1., 4.);
            *self.foodgrid[w][h].get_mut() += f;
        }

        // move blips
        let iter = self.blips.par_iter_mut();
        iter.for_each(|blip| blip.motion(args.dt));

        // update tree
        // todo: maybe this can be done smarter, instead of completely
        // rebuilding the tree it could be updated, keeping most of its structure
        let tree = self
            .blips
            .iter()
            .enumerate()
            .inspect(|(_, b)| {
                assert!(!b.status.pos[0].is_nan());
                assert!(!b.status.pos[1].is_nan())
            })
            .map(|(p, b)| BlipLoc::new(p, b.status.pos))
            .collect();
        self.tree = RTree::bulk_load(tree);
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
