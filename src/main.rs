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

mod vecmath;
use vecmath::Vector;

const FOOD_WIDTH: usize = 50;
const FOOD_HEIGHT: usize = 50;

const LOCAL_ENV: f64 = 500.;
const INITIAL_CELLS: usize = 200;

const SIM_WIDTH: f64 = (FOOD_WIDTH * 10) as f64;
const SIM_HEIGHT: f64 = (FOOD_HEIGHT * 10) as f64;

//todo: this should be a (mutating) parameter of the individual blips
const MUTATION_RATE: f64 = 0.01;

//todo: should be per blip
const REPR_TRES: f64 = 160.;

type BlipLoc = PointWithData<usize, [f64; 2]>;

pub struct App {
    gl: GlGraphics, // OpenGL drawing backend.
    // todo: maybe blips don't need to store their own positon
    blips: Vec<Blip>,
    // this probably needs to be atomic u64
    foodgrid: [[f64; FOOD_HEIGHT]; FOOD_WIDTH],
    tree: RTree<BlipLoc>,
    selection: Selection,
}

#[derive(Clone, Copy, Debug)]
enum Selection {
    None,
    Bigboy,
    Age,
    Young,
    Spawns,
}

impl Selection {
    // todo: add pre
    fn next(self) -> Self {
        match self {
            Selection::None => Selection::Bigboy,
            Selection::Bigboy => Selection::Age,
            Selection::Age => Selection::Young,
            Selection::Young => Selection::Spawns,
            Selection::Spawns => Selection::None,
        }
    }
}

// todo: this needs to be split
// outputs and inputs don't really need to be stored
// status might have to be split into genes and status
#[derive(Clone)]
struct Blip {
    /// things that change during the lifetime of a blip
    status: Status,
    /// things that only change trough mutation during reproduction
    genes: Genes,
}

impl Blip {
    fn split(&mut self) -> Self {
        self.status.hp /= 2.;
        self.status.children += 1;
        let mut new = self.clone();
        new.status.food = 0.;
        new.status.age = 0.;
        //todo: request rng
        let mut rng = rand::thread_rng();

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
}

#[derive(Clone)]
struct Genes {
    brain: SimpleBrain,
}

impl Genes {
    fn mutate<R: Rng>(&self, mut r: R) -> Self {
        let mut new = self.clone();
        new.brain.mutate(&mut r);
        new
    }
}

const N_INPUTS: usize = 2;

/// stored as an array for easy
/// neural network access.
/// but accessed/modified through methods
#[derive(Clone, PartialEq, Default)]
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
#[derive(Clone, PartialEq, Default)]
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
        self.data[2]
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
                let trans = self.foodgrid[w][h].min(10.) / 10.;
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
            let mut search = self.tree.locate_within_distance([px, py], LOCAL_ENV);
            let _ = search.next();
            let nb = search.next();
            if Some(index) == marker {
                polygon(PURPLE, TRI, transform.zoom(2.), gl);
            } else if nb.is_some() {
                polygon(RED, TRI, transform, gl);
            } else {
                polygon(BLUE, TRI, transform, gl);
            }
        }

        self.gl.draw_end();
    }

    fn update(&mut self, args: &UpdateArgs) {
        // perf: ask for rng as parameter
        let mut rng = rand::thread_rng();

        // update the inputs
        // todo: don't clone here, keep two buffers and swap instead
        let mut new = self.blips.clone();

        let mut spawns = Vec::new();

        // new is write only. if you need data from the last iteration
        // get it from old only.
        for (new, old) in new.iter_mut().zip(self.blips.iter()) {
            let mut inputs: Inputs = Default::default();

            let mut search = self.tree.locate_within_distance(old.status.pos, LOCAL_ENV);
            let _ = search.next();

            for nb in search {
                // sound
                let nb = &self.blips[nb.data];
                use rstar::PointDistance;
                let dist_squared = PointDistance::distance_2(&old.status.pos, &nb.status.pos);

                // todo: get rid of sqrt
                let nb_sound = (nb.status.vel[0] * nb.status.vel[0])
                    + (nb.status.vel[1] * nb.status.vel[1]).sqrt();

                *inputs.sound_mut() += nb_sound / dist_squared
            }

            let gridpos = [
                (new.status.pos[0] / 10.) as usize,
                (new.status.pos[1] / 10.) as usize,
            ];
            let gridpos = &mut self.foodgrid[gridpos[0]][gridpos[1]];
            *inputs.smell_mut() = *gridpos;

            // fixme: make sure dt is used literally on every change

            // inputs are processed at this point, time to feed the brain some data
            // todo: rn this accesses the new input, not the old, maybe thats good
            // maybe it needs to change

            let outputs = old.genes.brain.think(&inputs);
            // todo: maybe add some smoothing so outputs don't change instantly
            // otherwise storing them in the blip is pointless

            // now outputs are filled, time to act on them
            let spike = new.status.spike + outputs.spike() / 2.;
            new.status.spike = spike.min(1.).max(0.);

            // change direction
            const ORTHO: Vector = [0., 1.];
            let steer = vecmath::scale(ORTHO, outputs.steering());
            let dir = vecmath::norm(old.status.vel);

            let push = vecmath::rotate(steer, dir);

            let mut vel = vecmath::add(old.status.vel, push);
            // todo: be smarter about scaling speed, maybe do some log-stuff to simulate drag
            vel = vecmath::norm(vel);
            vel = vecmath::scale(vel, outputs.speed() * 50.);
            new.status.vel = vel;

            // eat food
            let speed = outputs.speed().abs() * 50.;
            let eating = (*gridpos / speed.max(1.)) * args.dt;
            if eating != 0. && !eating.is_nan() {
                *gridpos -= eating;
                new.status.food += eating;
            }
            let digest = new.status.food.min(10. * args.dt);
            new.status.food -= digest;
            new.status.hp += digest;

            // reproduce & mutate
            // reproduction is a bit of a problem since it needs to ad new entries to the vec
            // which is kinda bad for multithreading.
            // its a rather rare event though so its special-cased

            if new.status.hp > REPR_TRES {
                println!("new spawn!");
                let spawn = new.split();
                spawns.push(spawn);
            }
            new.status.age += args.dt;
        }

        new.extend(spawns);
        // todo: die, drop dead

        self.blips = new;

        // add some food
        // fixme: if dt > 10 this needs to run multiple times
        if rng.gen_bool(0.1 * args.dt) {
            let w: usize = rng.gen_range(0, FOOD_WIDTH);
            let h: usize = rng.gen_range(0, FOOD_HEIGHT);
            let f: f64 = rng.gen_range(1., 2.);

            self.foodgrid[w][h] += f;
        }

        // move blips
        for blip in &mut self.blips {
            let pos = &mut blip.status.pos;
            let delta = &mut blip.status.vel;
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
            .map(|(p, b)| BlipLoc::new(p, b.status.pos))
            .collect();
        self.tree = RTree::bulk_load(tree);
    }
}

#[derive(Copy, Clone, Default)]
struct SimpleBrain {
    // each output gets a weight for each input
    weights: [[f64; N_INPUTS]; N_OUTPUTS],
}
impl SimpleBrain {
    fn mutate<R: Rng>(&mut self, mut r: R) {
        for out in &mut self.weights {
            for inp in out.iter_mut() {
                *inp += r.gen_range(-0.1, 0.1) * r.gen_range(0., MUTATION_RATE);
                *inp *= 1. + (r.gen_range(-0.1, 0.1) * r.gen_range(0., MUTATION_RATE));
            }
        }
    }
}

impl SimpleBrain {
    fn init<R: Rng>(mut r: R) -> Self {
        let mut s: Self = Default::default();
        for out in &mut s.weights {
            for inp in out.iter_mut() {
                *inp = r.gen_range(-0.1, 0.1);
            }
        }
        s
    }
    fn think(&self, inputs: &Inputs) -> Outputs {
        let mut o: Outputs = Outputs::default();
        for (iw, o) in self.weights.iter().zip(o.data.iter_mut()) {
            let weighted_in: f64 = iw.iter().zip(&inputs.data).map(|(iw, i)| iw * i).sum();
            let clamped = weighted_in.max(-20.).min(20.);
            let res = 1. / (1. + (-clamped).exp());
            *o = res;
        }
        o
    }
}

fn main() {
    use atomic::Atomic;
    println!("f64 is atomic {}", Atomic::<f64>::is_lock_free());
    println!("f32 is atomic {}", Atomic::<f32>::is_lock_free());
    // Change this to OpenGL::V2_1 if not working.
    let opengl = OpenGL::V4_5;

    // Create an Glutin window.
    let mut window: Window = WindowSettings::new("pixelherd", [1000, 1000])
        .graphics_api(opengl)
        .exit_on_esc(true)
        .build()
        .unwrap();

    // Create a new game and run it.
    let mut app = App {
        gl: GlGraphics::new(opengl),
        blips: Vec::with_capacity(INITIAL_CELLS),
        tree: RTree::new(),
        foodgrid: [[0.; FOOD_HEIGHT]; FOOD_WIDTH],
        selection: Selection::None,
    };

    let mut rng = rand::thread_rng();

    for _ in 0..INITIAL_CELLS {
        let x = rng.gen_range(0., SIM_WIDTH);
        let y = rng.gen_range(0., SIM_HEIGHT);

        let dx = rng.gen_range(-30., 30.);
        let dy = rng.gen_range(-5., 5.);
        app.blips.push(Blip {
            status: Status {
                pos: [x, y],
                vel: [dx, dy],
                spike: 0.,
                hp: 100.,
                food: 50.,
                age: 0.,
                children: 0,
            },
            genes: Genes {
                brain: SimpleBrain::init(&mut rng),
            },
        });
    }

    for w in 0..FOOD_WIDTH {
        for h in 0..FOOD_HEIGHT {
            //fixme: this should be an exponential distribution instead
            if rng.gen_range(0, 10) == 1 {
                app.foodgrid[w][h] = rng.gen_range(0., 10.);
            }
        }
    }

    let mut events = Events::new(EventSettings::new());
    while let Some(e) = events.next(&mut window) {
        if let Some(args) = e.button_args() {
            if args.button == input::Button::Keyboard(input::keyboard::Key::A)
                && args.state == input::ButtonState::Release
            {
                app.selection = app.selection.next();
                println!("now highlighting {:?}", app.selection);
            }
        }
        if let Some(args) = e.render_args() {
            app.render(&args);
        }

        if let Some(args) = e.update_args() {
            app.update(&args);
        }
    }
}
