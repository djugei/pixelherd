use crate::config;
use rand::Rng;

use crate::brains;
use crate::brains::Brain;

use crate::app::FoodGrid;
use crate::app::OldFoodGrid;
use crate::app::TreeRef;

use crate::vecmath;
use crate::vecmath::Vector;

use crate::stablevec::StableVec;

use atomic::Ordering;

#[derive(Clone, Copy, PartialEq, Debug)]
// todo: as only status changes (and is kinda small) and genes do not change (and are kinad big
// cause of the brain) it might be smart to separate them, so the genes do not need to be copied on
// every execution
pub struct Blip<B: Brain + Clone> {
    /// things that change during the lifetime of a blip
    pub status: Status,
    /// things that only change trough mutation during reproduction
    pub genes: Genes<B>,
}

impl<B: Brain> Blip<B> {
    pub fn update<R: Rng>(
        &mut self,
        mut rng: R,
        old: &Self,
        olds: &StableVec<Self>,
        tree: TreeRef,
        oldgrid: &OldFoodGrid,
        foodgrid: &FoodGrid,
        time: f64,
        dt: f64,
    ) -> Option<Self> {
        let mut inputs: brains::Inputs = Default::default();

        let search = tree.query_distance(&self.status.pos, config::b::LOCAL_ENV);

        let base_angle = base_angle(&self.status);
        let mut eyedists = [(f64::INFINITY, 0., 0); config::b::N_EYES];
        let mut eye_angles = [0.; config::b::N_EYES];
        for i in 0..config::b::N_EYES {
            eye_angles[i] = eye_angle(base_angle, self.genes.eyes[i]);
        }

        for (dist_squared, (_p, index)) in search {
            // sound
            let nb = &olds.get(*index).unwrap();

            // todo: get rid of sqrt
            let nb_sound = (nb.status.vel[0] * nb.status.vel[0])
                + (nb.status.vel[1] * nb.status.vel[1]).sqrt();

            *inputs.sound_mut() += nb_sound / dist_squared;

            for (i, eye) in eye_angles.iter().enumerate() {
                let fov = 0.2 * std::f64::consts::PI;
                let angle = eye_vision(&self.status, *eye, nb.status.pos);
                // see the closest one in fov
                if angle.abs() < fov && eyedists[i].0 > dist_squared {
                    eyedists[i] = (dist_squared, angle, *index);
                }
            }
            for (&(dis, angle, id), inp) in eyedists.iter().zip(inputs.eyes_mut().iter_mut()) {
                if dis != f64::INFINITY {
                    let nb = &olds.get(id).unwrap();
                    let rgb = nb.status.rgb;
                    let write = [
                        rgb[0] as f64 - 0.5,
                        rgb[1] as f64 - 0.5,
                        rgb[2] as f64 - 0.5,
                        angle,
                        (dis / config::b::LOCAL_ENV) - 0.5,
                    ];
                    for (i, w) in inp.iter_mut().zip(&write) {
                        *i = *w;
                    }
                }
            }
        }

        // rust apparently does the modulo [-pi/2, pi/2] internally
        *inputs.clock1_mut() = (time * old.genes.clockstretch_1).sin();
        *inputs.clock2_mut() = (time * old.genes.clockstretch_2).sin();

        let gridpos = [
            (self.status.pos[0] / 10.) as usize,
            (self.status.pos[1] / 10.) as usize,
        ];
        let grid_slot = &foodgrid[gridpos[0]][gridpos[1]];

        let mut grid_value = oldgrid[gridpos[0]][gridpos[1]];

        //todo: also smell distance from center of tile
        // food is ~ 0-10+, scale to -5 to 5+
        *inputs.smell_mut() = grid_value - 5.;

        // inputs are processed at this point, time to feed the brain some data
        let outputs = old.genes.brain.think(&inputs);

        // eat food
        //todo: put into config
        // can eat 10 food / second (arbitrary)
        let max = 10. * dt;
        // half consumption speed on basically empty square
        // full consumption on 5 food, double on 15
        let gridfactor = 0.5 + (grid_value / 10.);
        // 1..11
        let div = 1. + (outputs.speed() * 2.5);
        let consumption = (max * gridfactor / div).min(grid_value);
        if consumption > 0. && !consumption.is_nan() {
            // retry writing the delta
            // todo: rename variables for clarity
            loop {
                let newval = grid_value - consumption;
                // there is no synchronization between threads, only the global food object
                // so only the atomic operaton on it needs to be taken care of.
                // there is no other operations to synchronize
                // relaxed ordering should therefore be fine to my best knowledge
                match grid_slot.compare_exchange(
                    grid_value,
                    newval,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        self.status.food += consumption;
                        break;
                    }
                    Err(v) => {
                        //println!("{:?} had to reloop for cas", gridpos);
                        grid_value = v;
                        continue;
                    }
                }
            }
        }

        // can only digest 2 out of the 10 theoretical max consumption food
        let digest = self.status.food.min(2. * dt);
        self.status.food -= digest;
        self.status.hp += digest;

        self.status.rgb = [
            outputs.r() as f32,
            outputs.g() as f32,
            outputs.b() as f32,
            1.,
        ];

        // now outputs are filled, time to act on them
        let spike = self.status.spike + outputs.spike() / 2.;
        self.status.spike = spike.min(1.).max(0.);

        // change direction
        let steer = [0., outputs.steering()];
        // fixme: handle 0 speed
        let dir = vecmath::norm(old.status.vel);

        let push = vecmath::rotate(steer, dir);

        let mut vel = vecmath::add(old.status.vel, push);
        // todo: be smarter about scaling speed, maybe do some log-stuff to simulate drag
        vel = vecmath::norm(vel);
        vel = vecmath::scale(vel, outputs.speed());
        self.status.vel = vel;

        // use energy

        let idling = 1.;
        // speed is (0..10)
        let movement = outputs.speed() / 10.;
        let ratiod =
            (idling * config::b::IDLE_E_RATIO) + (movement * (1. - config::b::IDLE_E_RATIO));
        let exhaustion = ratiod * dt * config::b::E_DRAIN;
        self.status.hp -= exhaustion;

        // reproduce & mutate
        // reproduction is a bit of a problem since it needs to ad new entries to the vec
        // which is kinda bad for multithreading.
        // its a rather rare event though so its special-cased

        let ret = if self.status.hp > old.genes.repr_tres {
            let spawn = self.split(&mut rng);
            Some(spawn)
        } else {
            None
        };
        self.status.age += dt;
        ret
    }

    /// move the blip according to its velocity, wrapping around the world
    /// generally called after update()
    pub fn motion(&mut self, dt: f64) {
        let pos = &mut self.status.pos;
        let delta = &mut self.status.vel;

        if delta[0].is_nan() || delta[1].is_nan() {
            dbg!(delta, pos);
            panic!();
        }
        pos[0] += delta[0] * dt;
        pos[1] += delta[1] * dt;

        pos[0] %= config::SIM_WIDTH;
        pos[1] %= config::SIM_HEIGHT;

        if pos[0] < 0. {
            pos[0] += config::SIM_WIDTH
        };
        if pos[1] < 0. {
            pos[1] += config::SIM_WIDTH
        };
    }
    pub fn new<R: Rng>(mut rng: R) -> Self {
        let x = rng.gen_range(0., config::SIM_WIDTH);
        let y = rng.gen_range(0., config::SIM_HEIGHT);

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
                rgb: [0.; 4],
            },
            genes: Genes::new(&mut rng),
        }
    }
    pub fn split<R: Rng>(&mut self, mut rng: R) -> Self {
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

#[derive(Copy, Clone, PartialEq, Default, Debug)]
pub struct Status {
    pub pos: [f64; 2],
    pub vel: [f64; 2],
    pub spike: f64,
    pub food: f64,
    pub hp: f64,
    pub age: f64,
    pub children: usize,
    pub generation: usize,
    pub rgb: [f32; 4],
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Genes<B: Brain> {
    pub brain: B,
    pub mutation_rate: f64,
    pub repr_tres: f64,
    // actual clock is multiplied by this
    pub clockstretch_1: f64,
    pub clockstretch_2: f64,
    // 3 eyes, each represented by an angle in radians [-pi-pi]
    pub eyes: [f64; config::b::N_EYES],
}

impl<B: Brain> Genes<B> {
    fn new<R: Rng>(mut rng: R) -> Self {
        use std::f64::consts::PI;
        Self {
            brain: B::init(&mut rng),
            mutation_rate: (rng.gen_range(-0.001, 0.001) + 0.01) * 4.,
            repr_tres: rng.gen_range(-10., 10.) + 100.,
            clockstretch_1: rng.gen_range(0.01, 1.),
            clockstretch_2: rng.gen_range(0.01, 1.),
            eyes: [
                rng.gen_range(-PI, PI),
                rng.gen_range(-PI, PI),
                rng.gen_range(-PI, PI),
            ],
            //eyes: [(-2. / 3.) * PI, 0., (2. / 3.) * PI],
            //eyes: [0., 0., 0.],
        }
    }

    fn mutate<R: Rng>(&self, mut rng: R) -> Self {
        //let mut new = self.clone();
        let mut brain = self.brain.clone();
        brain.mutate(&mut rng, self.mutation_rate);

        let repr_tres =
            self.repr_tres * (1. + rng.gen_range(-self.mutation_rate, self.mutation_rate));

        let mutation_rate =
            self.mutation_rate + (rng.gen_range(-self.mutation_rate, self.mutation_rate) / 10.);

        let clockstretch_1 =
            self.clockstretch_1 * (1. + rng.gen_range(-self.mutation_rate, self.mutation_rate));
        let clockstretch_2 =
            self.clockstretch_2 * (1. + rng.gen_range(-self.mutation_rate, self.mutation_rate));

        use std::f64::consts::PI;
        let mut eyes = self.eyes.clone();
        for eye in eyes.iter_mut() {
            *eye += rng.gen_range(-self.mutation_rate * PI, self.mutation_rate * PI);
            // wrap around
            *eye %= PI;
        }
        Self {
            brain,
            repr_tres,
            mutation_rate,
            clockstretch_1,
            clockstretch_2,
            eyes,
        }
    }
}
pub fn scaled_rand<R: Rng>(mut r: R, rate: f64, abs_scale: f64, mul_scale: f64, val: &mut f64) {
    *val += r.gen_range(-abs_scale, abs_scale) * r.gen_range(0., rate);
    *val *= 1. + (r.gen_range(-mul_scale, mul_scale) * r.gen_range(0., rate));
}

pub fn base_angle(s: &Status) -> f64 {
    vecmath::atan2(vecmath::norm(s.vel))
}

pub fn eye_angle(base_angle: f64, eye: f64) -> f64 {
    vecmath::rad_norm(base_angle + eye)
}

/// returns rad diff between the eye and the other blip
/// -pi-pi
pub fn eye_vision(
    status: &Status,
    //heading of the blip
    // absolute heading of the eye
    eye_angle: f64,
    other: Vector,
) -> f64 {
    let diff = vecmath::sub(other, status.pos);
    let diff_norm = vecmath::norm(diff);
    let angle = vecmath::atan2(diff_norm);
    let angle_diff = vecmath::rad_norm(angle - eye_angle);
    angle_diff
}
