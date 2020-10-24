// todo:
// 3. port one or both of the brains from c++
// 7. add state, both for pausing/resuming simulation and for extracting successful specimen
// 10. make sure distance calculations correctly work with the wraparound world (hint: rn they
//     don't)
// 11. balance simulation so scavenging and hunting are valid strategies

use opengl_graphics::{GlGraphics, OpenGL};
use sdl2_window::Sdl2Window as Window;
//use opengles_graphics::{GlGraphics, OpenGL};
use input::mouse::MouseCursorEvent;
use piston::event_loop::{EventSettings, Events};
use piston::input;
use piston::input::{ButtonEvent, RenderEvent, UpdateArgs, UpdateEvent};
use piston::window::WindowSettings;

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
    // Change this to OpenGL::V2_1 if not working.
    // fixme: i don't want to manually be guessing opengl versions
    let opengl = OpenGL::V4_5;

    // Create an Glutin window.
    let mut window: Window = WindowSettings::new("pixelherd", [1000, 1000])
        .graphics_api(opengl)
        .exit_on_esc(true)
        .build()
        .unwrap();

    let mut app = App::<brains::SimpleBrain>::new(1234, "report".into());
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
    // shouldbe building a rusttype::FontCollection and then select by index.
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
                        println!("spawning new blip currently disabled");
                        //fixme: app.blips.push(Blip::new(&mut rng));
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
                    app.update(&UpdateArgs { dt: 0.02 });
                }
            }
        }
    }
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
