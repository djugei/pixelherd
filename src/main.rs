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
use input::mouse::MouseCursorEvent;
use piston::event_loop::{EventSettings, Events};
use piston::input;
use piston::input::{ButtonEvent, RenderArgs, RenderEvent, UpdateArgs, UpdateEvent};
use piston::window::WindowSettings;

use rstar::primitives::PointWithData;
use rstar::RTree;

use rand::Rng;

use atomic::Atomic;
use rayon::prelude::IndexedParallelIterator;
use rayon::prelude::IntoParallelRefMutIterator;
use rayon::prelude::ParallelIterator;

#[allow(unused)]
use inline_tweak::tweak;

mod vecmath;
use vecmath::Vector;

mod brains;
use brains::BigBrain;

mod blip;
use blip::Blip;

mod select;
use select::Selection;

// coordinates:
// [width, height] <=> [x, y]
//
//        ^
//       -h
//
// <- -w  *  +w ->
//
//       +h
//        v
//
// so height is actually "depth"

const FOOD_WIDTH: usize = 50;
const FOOD_HEIGHT: usize = 50;

const LOCAL_ENV: f64 = 1500.;
const INITIAL_CELLS: usize = 100;
const REPLACEMENT: usize = 20;

const SIM_WIDTH: f64 = (FOOD_WIDTH * 10) as f64;
const SIM_HEIGHT: f64 = (FOOD_HEIGHT * 10) as f64;

type BlipLoc = PointWithData<usize, [f64; 2]>;
type FoodGrid = [[Atomic<f64>; FOOD_HEIGHT]; FOOD_WIDTH];

pub struct App {
    gl: GlGraphics, // OpenGL drawing backend.
    blips: Vec<Blip<BigBrain>>,
    // this probably needs to be atomic u64
    foodgrid: FoodGrid,
    tree: RTree<BlipLoc>,
    selection: Selection,
    time: f64,
    mousepos: [f64; 2],
}
/// displays multiline text
use graphics::types::Matrix2d;
fn display_text<C, G>(
    text: &str,
    glyph_cache: &mut C,
    // the left upper corner
    basetrans: Matrix2d,
    colour: [f32; 4],
    size: usize,
    graphics: &mut G,
) -> Result<(), <C as graphics::character::CharacterCache>::Error>
where
    G: graphics::Graphics,
    C: graphics::character::CharacterCache<Texture = G::Texture>,
    <C as graphics::character::CharacterCache>::Error: std::fmt::Debug,
{
    let basetrans = basetrans.trans(0., size as f64);
    use graphics::Transformed;
    text.split('\n')
        .enumerate()
        .map(|(idx, txt)| {
            graphics::text(
                colour,
                size as u32,
                txt,
                glyph_cache,
                basetrans.trans(0., (size * idx) as f64),
                graphics,
            )
        })
        .collect()
}

fn locate_in_radius(
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

impl App {
    fn render<C>(&mut self, args: &RenderArgs, glyph_cache: &mut C)
    where
        C: graphics::character::CharacterCache<Texture = opengl_graphics::Texture>,
        <C as graphics::character::CharacterCache>::Error: std::fmt::Debug,
    {
        use graphics::*;

        const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
        const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
        const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
        const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
        const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

        //const PURPLE: [f32; 4] = [1.0, 0.0, 1.0, 1.0];

        const TRI: types::Polygon = &[[-5., 0.], [0., -20.], [5., 0.]];
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
        let sim_x = self.mousepos[0] * SIM_WIDTH / width;
        let sim_y = self.mousepos[1] * SIM_HEIGHT / height;
        let marker = self
            .selection
            .select(&self.blips, &self.tree, &[sim_x, sim_y]);
        for (index, blip) in self.blips.iter().enumerate() {
            let (px, py) = (blip.status.pos[0], blip.status.pos[1]);
            let (pdx, pdy) = (blip.status.vel[0], blip.status.vel[1]);
            let pos_transform = c
                .transform
                .trans(px / SIM_WIDTH * width, py / SIM_HEIGHT * height);
            let transform = pos_transform.orient(pdx, pdy).rot_deg(90.);
            if Some(index) == marker {
                polygon(blip.status.rgb, TRI, transform.zoom(2.), gl);

                let base_angle = vecmath::atan2(vecmath::norm(blip.status.vel));

                let search = locate_in_radius(&self.tree, blip.status.pos, LOCAL_ENV)
                    .filter(|(p, _)| *p.position() != blip.status.pos)
                    .collect::<Vec<_>>();

                const RECT: [f64; 4] = [-5., -5., 10., 10.];
                use std::f64::consts::PI;
                let col = [RED, GREEN, BLUE];

                for (eye, col) in blip.genes.eyes.iter().zip(col.iter()) {
                    let eyesearch =
                        blip::eyefilter(search.iter(), &blip.status, *eye, 0.2 * PI, |(p, _d)| {
                            *p.position()
                        });

                    for (p, _) in eyesearch {
                        let p = *p.position();
                        let t = c
                            .transform
                            .trans(p[0] / SIM_WIDTH * width, p[1] / SIM_HEIGHT * height);

                        ellipse(*col, RECT, t, gl);
                    }
                }

                for (eye, col) in blip.genes.eyes.iter().zip(col.iter()) {
                    let (s, c) = eye.sin_cos();
                    line_from_to(
                        *col,
                        1.,
                        [0., 0.],
                        [s * 10., c * 10.],
                        transform.rot_rad(-PI),
                        gl,
                    );
                }
                let display = format!(
                    "children: {}\nhp: {:.2}\ngeneration: {}\nage: {:.2}\nheading: {:.2}\neyes: {:.2?}",
                    blip.status.children,
                    blip.status.hp,
                    blip.status.generation,
                    blip.status.age,
                    base_angle,
                    blip.genes.eyes,
                );
                let size = 20_usize;
                display_text(&display, glyph_cache, pos_transform, BLACK, size, gl).unwrap();
            } else {
                polygon(blip.status.rgb, TRI, transform, gl);
            }
            if blip.status.spike > 0.3 {
                let start = -20. * (2. / 3.);
                line_from_to(
                    RED,
                    1.2,
                    [0., start],
                    [0., start - (blip.status.spike * 10.)],
                    transform,
                    gl,
                );
            }
        }
        self.gl.draw_end();
    }

    // fixme: make sure dt is used literally on every change
    fn update<R: Rng>(&mut self, args: &UpdateArgs, mut rng: R) {
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
        selection: Selection::new(),
        time: 0.,
        mousepos: [0.; 2],
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
    let ts = opengl_graphics::TextureSettings::new();
    let mut cache =
    //fixme: choose a font thats actually available on systems
        opengl_graphics::GlyphCache::new("/usr/share/fonts/TTF/FiraCode-Regular.ttf", (), ts)
            .unwrap();

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
        if let Some(args) = e.mouse_cursor_args() {
            app.mousepos = args;
        }
        if let Some(args) = e.render_args() {
            if !hide {
                app.render(&args, &mut cache);
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
