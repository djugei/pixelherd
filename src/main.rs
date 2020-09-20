// todo:
// (1). create fixed-size (array) rtree instead of vec-based
//	(can mostly copy the rtree/spade crate, removing abstractions)
//	(pushing that back for now, its actually an ok implentation, especially for 100%safe)
// [2]. open a drawable window (conrod)
// [2.1.] draw a triangle
// 3. port one or both of the brains from c++
// 4. port the main game loop
// 5. parallelize it
// [6.] use rtree for distance lookups
// 7. add state, both for pausing/resuming simulation and for extracting successful specimen

use opengl_graphics::{GlGraphics, OpenGL};
use sdl2_window::Sdl2Window as Window;
//use opengles_graphics::{GlGraphics, OpenGL};
use piston::event_loop::{EventSettings, Events};
use piston::input;
use piston::input::{ButtonEvent, RenderArgs, RenderEvent, UpdateArgs, UpdateEvent};
use piston::window::WindowSettings;

use rstar::primitives::PointWithData;
use rstar::RTree;

use rand::Rng;

use atomic::Atomic;
use atomic::Ordering;
use rayon::prelude::IndexedParallelIterator;
use rayon::prelude::IntoParallelRefMutIterator;
use rayon::prelude::ParallelIterator;

mod vecmath;
use vecmath::Vector;

mod brains;
//use brains::SimpleBrain;
use brains::BigBrain as SimpleBrain;

const FOOD_WIDTH: usize = 50;
const FOOD_HEIGHT: usize = 50;

const LOCAL_ENV: f64 = 500.;
const INITIAL_CELLS: usize = 200;
const REPLACEMENT: usize = 20;

const SIM_WIDTH: f64 = (FOOD_WIDTH * 10) as f64;
const SIM_HEIGHT: f64 = (FOOD_HEIGHT * 10) as f64;

type BlipLoc = PointWithData<usize, [f64; 2]>;

pub struct App {
    gl: GlGraphics, // OpenGL drawing backend.
    blips: Vec<Blip>,
    // this probably needs to be atomic u64
    foodgrid: [[Atomic<f64>; FOOD_HEIGHT]; FOOD_WIDTH],
    tree: RTree<BlipLoc>,
    selection: Selection,
    time: f64,
}

#[derive(Clone, Copy, Debug)]
enum Selection {
    None,
    Bigboy,
    Age,
    Young,
    Spawns,
    Generation,
}

impl Selection {
    // todo: add pre
    fn next(self) -> Self {
        match self {
            Selection::None => Selection::Bigboy,
            Selection::Bigboy => Selection::Age,
            Selection::Age => Selection::Young,
            Selection::Young => Selection::Spawns,
            Selection::Spawns => Selection::Generation,
            Selection::Generation => Selection::None,
        }
    }
}

#[derive(Clone, PartialEq)]
struct Blip {
    /// things that change during the lifetime of a blip
    status: Status,
    /// things that only change trough mutation during reproduction
    genes: Genes,
}

impl Blip {
    fn new<R: Rng>(mut rng: R) -> Self {
        let x = rng.gen_range(0., SIM_WIDTH);
        let y = rng.gen_range(0., SIM_HEIGHT);

        let dx = rng.gen_range(-30., 30.);
        let dy = rng.gen_range(-5., 5.);
        Self {
            status: Status {
                pos: [x, y],
                vel: [dx, dy],
                spike: 0.,
                hp: 25.,
                food: 5.,
                age: 0.,
                children: 0,
                generation: 0,
            },
            genes: Genes::new(&mut rng),
        }
    }
    fn split<R: Rng>(&mut self, mut rng: R) -> Self {
        self.status.hp /= 2.;
        self.status.children += 1;
        let mut new = self.clone();
        new.status.generation += 1;
        new.status.food = 0.;
        new.status.age = 0.;
        new.status.children = 0;
        new.status.vel[0] += 1.;
        new.status.vel[1] += 1.;

        new.genes.mutate(&mut rng);
        new
    }
}

#[derive(Clone, PartialEq, Default)]
struct Status {
    pos: [f64; 2],
    vel: [f64; 2],
    spike: f64,
    food: f64,
    hp: f64,
    age: f64,
    children: usize,
    generation: usize,
}

#[derive(Clone, PartialEq)]
struct Genes {
    brain: SimpleBrain,
    mutation_rate: f64,
    repr_tres: f64,
}

impl Genes {
    fn new<R: Rng>(mut rng: R) -> Self {
        Self {
            brain: SimpleBrain::init(&mut rng),
            mutation_rate: rng.gen_range(-0.001, 0.001) + 0.01,
            repr_tres: rng.gen_range(-10., 10.) + 100.,
        }
    }
    fn mutate<R: Rng>(&self, mut rng: R) -> Self {
        let mut new = self.clone();
        let () = new.brain.mutate(&mut rng, self.mutation_rate);
        new.repr_tres *= 1. + rng.gen_range(-self.mutation_rate, self.mutation_rate);
        new.mutation_rate += rng.gen_range(-self.mutation_rate, self.mutation_rate) / 10.;
        new
    }
}

const N_INPUTS: usize = 2;

/// stored as an array for easy
/// neural network access.
/// but accessed/modified through methods
#[derive(Clone, PartialEq, Default, Debug)]
struct Inputs {
    data: [f64; N_INPUTS],
}

impl Inputs {
    //todo: add clocks as input
    pub fn sound_mut(&mut self) -> &mut f64 {
        &mut self.data[0]
    }
    pub fn smell_mut(&mut self) -> &mut f64 {
        &mut self.data[1]
    }
}

const N_OUTPUTS: usize = 4;

/// stored as an array for easy
/// neural network access.
/// but accessed/modified through methods
#[derive(Clone, PartialEq, Default, Debug)]
struct Outputs {
    data: [f64; N_OUTPUTS],
}
impl Outputs {
    pub fn spike(&self) -> f64 {
        self.data[0]
    }
    pub fn steering(&self) -> f64 {
        self.data[1]
    }
    pub fn speed(&self) -> f64 {
        self.data[2] * 1000.
    }
}

impl App {
    fn render(&mut self, args: &RenderArgs) {
        use graphics::*;

        const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
        //const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
        const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
        const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

        const PURPLE: [f32; 4] = [1.0, 0.0, 1.0, 1.0];

        const TRI: types::Polygon = &[[-0.5, 0.], [0., -2.], [0.5, 0.]];
        const SQR: types::Rectangle = [0., 0., 1., 1.];

        let (width, height) = (args.window_size[0], args.window_size[1]);

        let c = self.gl.draw_begin(args.viewport());
        let gl = &mut self.gl;
        clear(WHITE, gl);
        for w in 0..FOOD_WIDTH {
            for h in 0..FOOD_HEIGHT {
                let transform = c
                    .transform
                    .trans(
                        (w * 10) as f64 / SIM_WIDTH * width,
                        (h * 10) as f64 / SIM_HEIGHT * height,
                    )
                    .zoom(10.)
                    .scale(width / SIM_WIDTH, height / SIM_HEIGHT);
                // maybe logscale this
                let trans = self.foodgrid[w][h].get_mut().min(10.) / 10.;
                rectangle([0.0, 1.0, 0.0, trans as f32], SQR, transform, gl);
            }
        }
        let mut marker = None;
        match self.selection {
            Selection::None => marker = None,
            Selection::Bigboy => {
                let choice = self
                    .blips
                    .iter()
                    .enumerate()
                    .map(|(i, b)| (i, b.status.hp + b.status.food))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                if let Some(choice) = choice {
                    marker = Some(choice.0);
                }
            }
            Selection::Age => {
                let choice = self
                    .blips
                    .iter()
                    .enumerate()
                    .map(|(i, b)| (i, b.status.age))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                if let Some(choice) = choice {
                    marker = Some(choice.0);
                }
            }
            Selection::Young => {
                let choice = self
                    .blips
                    .iter()
                    .enumerate()
                    .map(|(i, b)| (i, b.status.age))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                if let Some(choice) = choice {
                    marker = Some(choice.0);
                }
            }
            Selection::Spawns => {
                let choice = self
                    .blips
                    .iter()
                    .enumerate()
                    .map(|(i, b)| (i, b.status.children))
                    .max_by(|a, b| a.1.cmp(&b.1));
                if let Some(choice) = choice {
                    marker = Some(choice.0);
                }
            }
            Selection::Generation => {
                let choice = self
                    .blips
                    .iter()
                    .enumerate()
                    .map(|(i, b)| (i, b.status.generation))
                    .max_by(|a, b| a.1.cmp(&b.1));
                if let Some(choice) = choice {
                    marker = Some(choice.0);
                }
            }
        }
        for (index, blip) in self.blips.iter().enumerate() {
            let (px, py) = (blip.status.pos[0], blip.status.pos[1]);
            let (pdx, pdy) = (blip.status.vel[0], blip.status.vel[1]);
            let transform = c
                .transform
                .trans(px / SIM_WIDTH * width, py / SIM_HEIGHT * height)
                .zoom(7.)
                .orient(pdx, pdy)
                .rot_deg(90.);
            let mut search = self
                .tree
                .locate_within_distance([px, py], LOCAL_ENV)
                .filter(|d| d.position() != &[px, py]);
            let nb = search.next();
            if Some(index) == marker {
                //todo: text-render those stats
                /*println!(
                    "repr {}, chidlren: {}, hp: {}, food: {}",
                    blip.genes.repr_tres, blip.status.children, blip.status.hp, blip.status.food
                );*/
                polygon(PURPLE, TRI, transform.zoom(2.), gl);
            } else if nb.is_some() {
                polygon(RED, TRI, transform, gl);
            } else {
                polygon(BLUE, TRI, transform, gl);
            }
        }

        self.gl.draw_end();
    }

    fn update<R: Rng>(&mut self, args: &UpdateArgs, mut rng: R) {
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
            let mut inputs: Inputs = Default::default();

            let search = self
                .tree
                .locate_within_distance(old.status.pos, LOCAL_ENV)
                .filter(|d| d.position() != &old.status.pos);

            for nb in search {
                // sound
                let nb = &self.blips[nb.data];
                use rstar::PointDistance;
                let dist_squared = PointDistance::distance_2(&old.status.pos, &nb.status.pos);

                // todo: get rid of sqrt
                let nb_sound = (nb.status.vel[0] * nb.status.vel[0])
                    + (nb.status.vel[1] * nb.status.vel[1]).sqrt();

                *inputs.sound_mut() += nb_sound / dist_squared;
                if inputs.sound_mut().is_infinite() {
                    dbg!(
                        nb_sound,
                        dist_squared,
                        nb.status.vel,
                        nb.status.pos,
                        old.status.vel,
                        old.status.pos,
                    );
                    panic!();
                }
            }

            let gridpos = [
                (new.status.pos[0] / 10.) as usize,
                (new.status.pos[1] / 10.) as usize,
            ];
            let grid_slot = &self.foodgrid[gridpos[0]][gridpos[1]];

            let casstat = Atomic::new(0usize);

            // there is no synchronization between threads, only the global food object
            // so only the atomic operaton on it needs to be taken care of.
            // there is no other operations to synchronize
            // relaxed ordering should therefore be fine to my best knowledge
            let mut grid_value = grid_slot.load(Ordering::Relaxed);
            let mut outputs;
            loop {
                *inputs.smell_mut() = grid_value;

                // fixme: make sure dt is used literally on every change

                // inputs are processed at this point, time to feed the brain some data
                // todo: rn this accesses the new input, not the old, maybe thats good
                // maybe it needs to change

                outputs = old.genes.brain.think(&inputs);

                // eat food
                //todo: consider using actual speed, not acceleration
                let speed = outputs.speed().abs();
                let eating = (grid_value / speed.max(1.)) * args.dt;
                if eating != 0. && !eating.is_nan() {
                    let newval = grid_value - eating;
                    match grid_slot.compare_exchange(
                        grid_value,
                        newval,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            new.status.food += eating;
                        }
                        Err(v) => {
                            //println!("{:?} had to reloop for cas", gridpos);
                            casstat.fetch_add(1, Ordering::Relaxed);
                            grid_value = v;
                            continue;
                        }
                    }
                }
                break;
            }
            let casstat = casstat.into_inner();
            if casstat > 1 {
                println!("[{}]: had to cas-loop {} times", self.time, casstat);
            }
            let digest = new.status.food.min(4. * args.dt);
            new.status.food -= digest;
            new.status.hp += digest;

            // now outputs are filled, time to act on them
            let spike = new.status.spike + outputs.spike() / 2.;
            new.status.spike = spike.min(1.).max(0.);

            // change direction
            const ORTHO: Vector = [0., 1.];
            let steer = vecmath::scale(ORTHO, outputs.steering());
            // fixme: handle 0 speed
            let dir = vecmath::norm(old.status.vel);

            let push = vecmath::rotate(steer, dir);

            let mut vel = vecmath::add(old.status.vel, push);
            // todo: be smarter about scaling speed, maybe do some log-stuff to simulate drag
            vel = vecmath::norm(vel);
            vel = vecmath::scale(vel, outputs.speed());
            new.status.vel = vel;

            // use energy

            let exhaustion = (0.1 + (outputs.speed().abs() * 0.4)) * args.dt * 0.1;
            new.status.hp -= exhaustion;

            // reproduce & mutate
            // reproduction is a bit of a problem since it needs to ad new entries to the vec
            // which is kinda bad for multithreading.
            // its a rather rare event though so its special-cased

            if new.status.hp > old.genes.repr_tres {
                println!("new spawn!");
                let spawn = new.split(&mut rng);
                let mut guard = spawns.lock().unwrap();
                guard.push(spawn);
            }
            new.status.age += args.dt;
        });

        let spawns = spawns.into_inner().unwrap();

        new.extend(spawns);
        // todo: die, drop dead

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
        for blip in &mut self.blips {
            let pos = &mut blip.status.pos;
            let delta = &mut blip.status.vel;

            if delta[0].is_nan() || delta[1].is_nan() {
                dbg!(delta, pos);
                panic!();
            }
            pos[0] += delta[0] * args.dt;
            pos[1] += delta[1] * args.dt;

            pos[0] %= SIM_WIDTH;
            pos[1] %= SIM_HEIGHT;

            if pos[0] < 0. {
                pos[0] += SIM_WIDTH
            };
            if pos[1] < 0. {
                pos[1] += SIM_WIDTH
            };
        }

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

fn scaled_rand<R: Rng>(mut r: R, rate: f64, abs_scale: f64, mul_scale: f64, val: &mut f64) {
    *val += r.gen_range(-abs_scale, abs_scale) * r.gen_range(0., rate);
    *val *= 1. + (r.gen_range(-mul_scale, mul_scale) * r.gen_range(0., rate));
}

fn main() {
    // Change this to OpenGL::V2_1 if not working.
    let opengl = OpenGL::V4_5;

    // Create an Glutin window.
    let mut window: Window = WindowSettings::new("pixelherd", [1000, 1000])
        .graphics_api(opengl)
        .exit_on_esc(true)
        .build()
        .unwrap();

    let foodgrid = [[0.; FOOD_HEIGHT]; FOOD_WIDTH];
    //safety: this is absolutely not safe as i am relying on the internal memory layout of a third
    // party library that is almost guaranteed to not match on 32 bit platforms.
    //
    // however i see no other way to initialize this array
    // try_from is only implemented for array up to size 32 because fucking rust has not const
    // generics
    // atomics are not copy, so the [0.;times] constructor does not work
    // this is an actual value, not a reference so i need to actually change the value instead of
    // "as-casting" the pointer
    let foodgrid = unsafe { std::mem::transmute(foodgrid) };

    // Create a new game and run it.
    let mut app = App {
        gl: GlGraphics::new(opengl),
        blips: Vec::with_capacity(INITIAL_CELLS),
        tree: RTree::new(),
        foodgrid,
        selection: Selection::None,
        time: 0.,
    };

    let mut rng = rand::thread_rng();

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

    let mut hurry = 1;
    let mut pause = false;
    let mut hide = false;

    let mut events = Events::new(EventSettings::new());
    while let Some(e) = events.next(&mut window) {
        if let Some(args) = e.button_args() {
            match args.button {
                input::Button::Keyboard(input::keyboard::Key::A) => {
                    if args.state == input::ButtonState::Release {
                        app.selection = app.selection.next();
                        println!("now highlighting {:?}", app.selection);
                    }
                }
                input::Button::Keyboard(input::keyboard::Key::S) => {
                    if args.state == input::ButtonState::Release {
                        println!("spawning new blip");
                        app.blips.push(Blip::new(&mut rng));
                    }
                }
                input::Button::Keyboard(input::keyboard::Key::NumPadPlus) => {
                    if args.state == input::ButtonState::Release {
                        hurry += 1;
                        println!("now running {} 0.02 updates per update", hurry);
                    }
                }
                input::Button::Keyboard(input::keyboard::Key::NumPadMinus) => {
                    if args.state == input::ButtonState::Release {
                        if hurry > 1 {
                            hurry -= 1;
                        }
                        println!("now running {} 0.02 updates per update", hurry);
                    }
                }
                input::Button::Keyboard(input::keyboard::Key::Space) => {
                    if args.state == input::ButtonState::Release {
                        pause = !pause;
                        println!("pausing {}", pause);
                    }
                }
                input::Button::Keyboard(input::keyboard::Key::H) => {
                    if args.state == input::ButtonState::Release {
                        hide = !hide;
                        println!("hiding rendering {}", pause);
                    }
                }
                input::Button::Keyboard(k) => {
                    println!("unhandled keypress: {:?} ({:?})", k, args.button);
                }
                input::Button::Mouse(_) => (),
                input::Button::Controller(_) => (),
                input::Button::Hat(_) => (),
            }
        }
        if let Some(args) = e.render_args() {
            if !hide {
                app.render(&args);
            }
        }

        if let Some(args) = e.update_args() {
            if pause {
            } else {
                if hide {
                    for _ in 0..1000 {
                        app.update(&UpdateArgs { dt: 0.02 }, &mut rng);
                    }
                } else {
                    if hurry == 1 {
                        app.update(&args, &mut rng);
                    } else {
                        for _ in 0..hurry {
                            app.update(&UpdateArgs { dt: 0.02 }, &mut rng);
                        }
                    }
                }
            }
        }
    }
}
