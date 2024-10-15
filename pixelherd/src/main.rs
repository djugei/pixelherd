// todo:
// 3. port one or both of the brains from c++
// 7. enable extraction of successful specimen
// 10. make sure distance calculations correctly work with the wraparound world (hint: rn they
//     don't)
// 11. balance simulation so scavenging and hunting are valid strategies
// 12. re-think speed, it currently does not play nice with brain output, "neutral" output is forward currently
// 13. introduce genetics

//! Pixelherd is a deterministic high performance evolutionary animal simulation
//!
//! Lets go over this step by step from the back.
//!
//! ## Animal Simulation
//! The simulated world in pixelherd is a grid,
//! each slot in the grid has a certain amount of vegetable and meat food.
//!
//! In this world there exist a number of simple animals, called "blips" in the code.
//! They can sense a lot of things from their environment and about themselves,
//! like how much food is at their current location, how healthy they themselves are, etc.
//!
//! They can also sense things about other blips, which colour they are and how fast blips
//! around them are moving for example.
//!
//! They can not only sense things, they can also interact with the world and other blips,
//! mainly through two means: They can control where they are going and how fast,
//! and they can extend a spike to ram into others. They can also change colour which can be
//! sensed by other blips.
//!
//! The inputs and the outputs are connected by a brain, a simple artificial neural network.
//! It gets fed with all the inputs and calculates what it considers appropriate outputs in reaction.
//! It even has a bit of memory and a sense of time.
//! Now, unlike a lot of neural networks, this one is not trained, instead it is evolved,
//! which leads to the next keyword.
//!
//! ## Evolutionary
//! Blips reproduce when they have enough food, creating an almost identical copy of themselves.
//! Some of the properties of the child are randomly mutated.
//! If the changes are beneficial then the child will reproduce more on average, which also spreads
//! the mutation.
//!
//! Note that at least right now pixelherd is not a genetic simulation, there is no gene pool,
//! no sexual reproduction. There might be in the future as this would allow single genes to
//! proliferate instead of having to drag around all the entire genetic information they happen to
//! be around at time of mutation. It could also lead to interesting courtship behaviour.
//!
//! ## High Performance
//! High performance in this case does not refer to the code being very well optimized or
//! the simulation trying to take some kind of shortcuts. Instead it refers to its ability to utilize
//! a large amounts of parallel processors and only having very little code that needs to be executed
//! in series.
//!
//! ## Deterministic
//! This one is rather simple:
//! When re-executed with the same seed and configuration the simulation will lead to the same results.
//!
//! Or rather: it would be if not for the parallel execution.
//! No matter in which order things are executed, the results must stay the same.
//!
//! # Hacking
//! The code is meant to be hacked, modified, played around with.
//! Open main.rs to read the non-doc comments about how the above properties are implemented
//! and how the project is structured.

// Welcome to the *code* :)
// The main fun is the App::update() function in the app module.
// It gets called by main() after doing some setup to display things and handle input.
// It can be called on its own though, if you just want results, no visualisation.
// Visualisation is handled by the Renderer struct in the renderer module.
// The render() function is probably the ugliest bit of code in here,
// so lets talk about the App::update() method instead:
//
// it does a bit of bookkeeping, then, in parallel, updates all blips (from the blip module)
// by calling the Blip::update() function.
// Afterwards it handles births and deaths, does a bit more bookkeeping
// and is ready for the next go around.
//
// The Blip::update() function has three parts.
// First it collects inputs, then it feeds them into its brain.
// Next it thinks using the brain.
// Finally it reads the Brains output and acts on them.
//
// There is more detail on how things work in the doc comments on the individual mentioned
// structs and methods.
// If you just want to quickly change some parameters have a look at the config module.
// Or you can read through main() and discover all key bindings.

use opengl_graphics::{GlGraphics, OpenGL};
use sdl2_window::Sdl2Window as Window;
//use opengles_graphics::{GlGraphics, OpenGL};
use input::mouse::MouseCursorEvent;
use piston::event_loop::{EventSettings, Events};
use piston::input;
use piston::input::{ButtonEvent, RenderEvent, UpdateEvent};
use piston::window::WindowSettings;

// this allows you to put values into a tweak!() macro, and modify them
// while the code is running.
// As in you literally put a different value into the source code, save, and it live-updates.
// Kinda crazy what these computers can do.
// This obviously breaks determinism though.
#[allow(unused)]
use inline_tweak::tweak;

mod config;

mod vecmath;

mod brains;

mod blip;

mod select;
use select::Selection;

mod stablevec;

mod app;
use app::App;

mod renderer;
use renderer::Renderer;

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
// todo: make sure this is actually true, something seems fucky

fn main() {
    // fixme: i don't want to manually be guessing opengl versions
    let opengl = OpenGL::V4_5;

    // Create an Glutin window.
    let mut window: Window = WindowSettings::new("pixelherd", [1000, 1000])
        .graphics_api(opengl)
        .exit_on_esc(true)
        .build()
        .unwrap();

    // try to restore previous state, if it exists.
    let oa = std::fs::File::open("savestate")
        .ok()
        .map(|f| App::new_from(f, "report".into()).ok())
        .flatten();
    if oa.is_some() {
        println!("loading from savestate")
    };
    // otherwise build a new simulation.
    let mut app: App<brains::SimpleBrain> = oa.unwrap_or_else(|| App::new(1234, "report".into()));

    // graphics stuff
    let mut render = Renderer {
        gl: GlGraphics::new(opengl),
        mousepos: [0.; 2],
        selection: Selection::new(),
    };

    let ts = opengl_graphics::TextureSettings::new();

    // font_kit is a bit "heavy" i only need font loading, could not really find a good other lib
    // for that though.
    use font_kit::family_name::FamilyName;
    use font_kit::handle::Handle;
    use font_kit::properties::Properties;
    use font_kit::source::SystemSource;

    let fontprops = Properties::new();
    let fontfam = [
        FamilyName::Title("FiraCode".to_owned()),
        FamilyName::SansSerif,
    ];
    let handle = SystemSource::new()
        .select_best_match(&fontfam, &fontprops)
        .unwrap();
    // fixme: this will (probably soft-) fail on font collections, as i am ignoring the index.
    // should be building a rusttype::FontCollection and then select by index.
    let fontdata: Result<std::path::PathBuf, Vec<u8>> = match handle {
        Handle::Path { path, .. } => Ok(path),
        Handle::Memory { bytes, .. } => Err((*bytes).clone()),
    };
    let mut cache = match fontdata.as_ref() {
        Ok(path) => {
            println!("using font: {:?}", path);
            opengl_graphics::GlyphCache::new(path, (), ts).unwrap()
        }
        Err(bytes) => opengl_graphics::GlyphCache::from_bytes(bytes, (), ts).unwrap(),
    };
    // end of graphics stuff

    // input and event handling.
    let mut speed = 1;
    let mut pause = false;
    let mut hyper = false;

    let mut events = Events::new(EventSettings::new());
    while let Some(e) = events.next(&mut window) {
        if let Some(args) = e.button_args() {
            match args.button {
                input::Button::Keyboard(input::keyboard::Key::A) => {
                    if args.state == input::ButtonState::Release {
                        render.selection = render.selection.rotate();
                        println!("now highlighting {:?}", render.selection);
                    }
                }
                input::Button::Keyboard(input::keyboard::Key::E) => {
                    if args.state == input::ButtonState::Release {
                        render.selection = render.selection.rotate_rev();
                        println!("now highlighting {:?}", render.selection);
                    }
                }
                input::Button::Keyboard(input::keyboard::Key::S) => {
                    if args.state == input::ButtonState::Release {
                        println!("saving state");
                        let mut f = std::fs::File::create("savestate").unwrap();
                        app.write_into(&mut f).unwrap();
                        println!("saved state");
                    }
                }
                input::Button::Keyboard(input::keyboard::Key::R) => {
                    if args.state == input::ButtonState::Release {
                        app.report()
                    }
                }
                input::Button::Keyboard(input::keyboard::Key::NumPadPlus) => {
                    if args.state == input::ButtonState::Release {
                        speed += 1;
                        println!(
                            "now running {} 0.02 updates per update",
                            if hyper { 1000 } else { 1 } * speed
                        );
                    }
                }
                input::Button::Keyboard(input::keyboard::Key::NumPadMinus) => {
                    if args.state == input::ButtonState::Release {
                        if speed > 1 {
                            speed -= 1;
                        }
                        println!(
                            "now running {} 0.02 updates per update",
                            if hyper { 1000 } else { 1 } * speed
                        );
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
                        hyper = !hyper;
                        if hyper {
                            println!("HYPERSPEED");
                        } else {
                            println!("regular speed");
                        }
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
            render.mousepos = args;
        }
        if let Some(args) = e.render_args() {
            //if !hide {
            render.render(&app, &args, &mut cache);
            //}
        }

        if e.update_args().is_some() {
            if pause {
            } else {
                let times = if hyper { 1000 * speed } else { speed };
                // always running fixed step now, for the determinism
                for _ in 0..times {
                    app.update();
                }
            }
        }
    }
    println!("exiting gracefully, saving state");
    let mut f = std::fs::File::create("savestate").unwrap();
    app.write_into(&mut f).unwrap();
    println!("goodbye!");
}

#[test]
fn boolsize() {
    assert_eq!(std::mem::size_of::<bool>(), 1);
}

#[test]
fn test_atomic() {
    use atomic::Atomic;
    use std::mem::{align_of, size_of};
    let base = (
        Atomic::<u64>::is_lock_free(),
        size_of::<u64>(),
        align_of::<u64>(),
    );
    let split1 = (
        Atomic::<(f32, f32)>::is_lock_free(),
        size_of::<(f32, f32)>(),
        align_of::<(f32, f32)>(),
    );
    let split2 = (
        Atomic::<[f32; 2]>::is_lock_free(),
        size_of::<[f32; 2]>(),
        align_of::<[f32; 2]>(),
    );
    #[repr(align(8))]
    struct Force {
        _food: f32,
        _meat: f32,
    }
    let force = (
        Atomic::<Force>::is_lock_free(),
        size_of::<Force>(),
        align_of::<Force>(),
    );
    // result: on architectures where 64-bit-atomics exist i can split them into 2 f32
    // by forcing alignment to be 8 (instead of the native 4)
    // i.e. (f32,f32) and [f32;2] do not work
    dbg!(base, split1, split2, force, Atomic::<i128>::is_lock_free());
    assert!(Atomic::<Force>::is_lock_free());
}

#[test]
#[ignore]
fn bench_brains() {
    // this is a poor mans benchmark because i didn't want to set up criterion for literally one
    // testcase and the std bench feature is nightly
    let times = 50_000;
    use std::time::Instant;

    use crate::brains::BigBrain;
    use crate::brains::SimpleBrain;

    let mut app1 = App::<SimpleBrain>::new(1234, None);
    let before = Instant::now();
    for _ in 0..times {
        app1.update(&UpdateArgs { dt: 0.02 });
    }
    let app1_t = before.elapsed().as_millis();

    let mut app2 = App::<Box<SimpleBrain>>::new(1234, None);
    let before = Instant::now();
    for _ in 0..times {
        app2.update(&UpdateArgs { dt: 0.02 });
    }
    let app2_t = before.elapsed().as_millis();

    let mut app3 = App::<BigBrain>::new(1234, None);
    let before = Instant::now();
    for _ in 0..times {
        app3.update(&UpdateArgs { dt: 0.02 });
    }
    let app3_t = before.elapsed().as_millis();

    let mut app4 = App::<Box<BigBrain>>::new(1234, None);
    let before = Instant::now();
    for _ in 0..times {
        app4.update(&UpdateArgs { dt: 0.02 });
    }
    let app4_t = before.elapsed().as_millis();

    println!("app1: {}mili", app1_t);
    println!("app2: {}mili", app2_t);

    println!("app1: {}mili", app3_t);
    println!("app2: {}mili", app4_t);
}
