use crate::app::locate_in_radius;
use crate::app::App;
use crate::blip;
use crate::config::*;
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
    pub fn render<C>(&mut self, app: &App, args: &RenderArgs, glyph_cache: &mut C)
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
                let trans = app.foodgrid()[w][h]
                    // relaxed ordering should be fine, nobody should really be accessing the app
                    // during render
                    .load(atomic::Ordering::Relaxed)
                    .min(10.)
                    / 10.;
                rectangle([0.0, 1.0, 0.0, trans as f32], SQR, transform, gl);
            }
        }
        // fixme move all coordinate calculations out
        // ideally i would have two types, simpos and screenpos
        let sim_x = self.mousepos[0] * SIM_WIDTH / width;
        let sim_y = self.mousepos[1] * SIM_HEIGHT / height;
        let marker = self
            .selection
            .select(app.blips(), app.tree(), &[sim_x, sim_y]);
        for (index, blip) in app.blips().iter().enumerate() {
            let (px, py) = (blip.status.pos[0], blip.status.pos[1]);
            let (pdx, pdy) = (blip.status.vel[0], blip.status.vel[1]);
            let pos_transform = c
                .transform
                .trans(px / SIM_WIDTH * width, py / SIM_HEIGHT * height);
            let transform = pos_transform.orient(pdx, pdy).rot_deg(90.);
            if Some(index) == marker {
                polygon(blip.status.rgb, TRI, transform.zoom(2.), gl);
                let base_angle = vecmath::atan2(vecmath::norm(blip.status.vel));

                let search = locate_in_radius(&app.tree(), blip.status.pos, LOCAL_ENV)
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
