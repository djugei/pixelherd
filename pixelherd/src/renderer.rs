use crate::app::App;
use crate::blip;
use crate::config;
use crate::select::Selection;
use crate::vecmath;
use crate::GlGraphics;
use piston::input::RenderArgs;

pub struct Renderer {
    pub gl: GlGraphics,
    pub mousepos: [f64; 2],
    pub selection: Selection,
}
impl Renderer {
    // todo: add brain visualisation
    pub fn render<C, B>(&mut self, app: &App<B>, args: &RenderArgs, glyph_cache: &mut C)
    where
        C: graphics::character::CharacterCache<Texture = opengl_graphics::Texture>,
        <C as graphics::character::CharacterCache>::Error: std::fmt::Debug,
        B: crate::brains::Brain + Send + Sync,
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
        clear(BLACK, gl);
        for w in 0..config::FOOD_WIDTH {
            for h in 0..config::FOOD_HEIGHT {
                let transform = c
                    .transform
                    .trans(
                        (w * 10) as f64 / config::SIM_WIDTH * width,
                        (h * 10) as f64 / config::SIM_HEIGHT * height,
                    )
                    .zoom(10.)
                    .scale(width / config::SIM_WIDTH, height / config::SIM_HEIGHT);
                // maybe logscale this
                let g = app.vegtables()[w][h]
                    // relaxed ordering should be fine, nobody should really be accessing the app
                    // during render (TM)
                    .load(atomic::Ordering::Relaxed)
                    .to_f64()
                    .min(10.)
                    / 10.;
                let r = app.meat()[w][h]
                    .load(atomic::Ordering::Relaxed)
                    .to_f64()
                    .min(10.)
                    / 10.;
                rectangle([r as f32, g as f32, 0.0, 1.0], SQR, transform, gl);
            }
        }
        // fixme move all coordinate calculations out
        // ideally i would have two types, simpos and screenpos
        let sim_x = self.mousepos[0] * config::SIM_WIDTH / width;
        let sim_y = self.mousepos[1] * config::SIM_HEIGHT / height;
        let marker = self
            .selection
            .select(app.blips_ro().enumerate(), app.tree(), &[sim_x, sim_y]);
        for (index, (status, genes)) in app.blips_ro().enumerate() {
            let (px, py) = (status.pos[0], status.pos[1]);
            let (pdx, pdy) = (status.vel[0], status.vel[1]);
            let pos_transform = c.transform.trans(
                px / config::SIM_WIDTH * width,
                py / config::SIM_HEIGHT * height,
            );
            let transform = pos_transform.orient(pdx, pdy).rot_deg(90.);
            if Some(index) == marker {
                polygon(status.rgb, TRI, transform.zoom(2.), gl);
                let tree = app.tree();
                let search = tree
                    .query_distance(&status.pos, config::b::LOCAL_ENV)
                    .collect::<Vec<_>>();

                const RECT: [f64; 4] = [-5., -5., 10., 10.];
                use std::f64::consts::PI;
                let col = [RED, GREEN, BLUE];

                let base_angle = blip::base_angle(&status);
                for (eye, col) in genes.eyes.iter().zip(col.iter()) {
                    let eye_angle = blip::eye_angle(base_angle, *eye);
                    let visual_fov = 0.2 * PI;
                    let eyesearch = search.iter().filter(|(_d, (p, _index))| {
                        blip::eye_vision(&status, eye_angle, *p).abs() < visual_fov
                    });

                    for (_d, (p, _index)) in eyesearch {
                        let t = c.transform.trans(
                            p[0] / config::SIM_WIDTH * width,
                            p[1] / config::SIM_HEIGHT * height,
                        );

                        ellipse(*col, RECT, t, gl);
                    }
                }

                for (eye, col) in genes.eyes.iter().zip(col.iter()) {
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
                    "children: {}\nhp: {:.2}\ngeneration: {}\nage: {:.2}\nheading: {:.2}\nspeed: {:.2?}\nmemory: {:.4?}\nvore: {:.2}",
                    status.children,
                    status.hp,
                    status.generation,
                    status.age,
                    base_angle,
                    vecmath::len(status.vel),
                    status.memory,
                    genes.vore,
                );
                let size = 20_usize;
                display_text(&display, glyph_cache, pos_transform, WHITE, size, gl).unwrap();
            } else {
                polygon(status.rgb, TRI, transform, gl);
            }
            if status.spike > 0.3 {
                let start = -20. * (2. / 3.);
                line_from_to(
                    RED,
                    1.2,
                    [0., start],
                    [0., start - (status.spike * 10.)],
                    transform,
                    gl,
                );
            }
        }
        self.gl.draw_end();
    }
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
    text.split('\n').enumerate().try_for_each(|(idx, txt)| {
        graphics::text(
            colour,
            size as u32,
            txt,
            glyph_cache,
            basetrans.trans(0., (size * idx) as f64),
            graphics,
        )
    })
}
