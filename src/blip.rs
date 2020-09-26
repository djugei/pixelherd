use crate::LOCAL_ENV;
use crate::SIM_HEIGHT;
use crate::SIM_WIDTH;
use rand::Rng;

use crate::brains;
use crate::brains::Brain;

use rstar::RTree;

use crate::BlipLoc;
use crate::FoodGrid;

use crate::vecmath;
use crate::vecmath::Vector;

use atomic::Atomic;
use atomic::Ordering;

#[allow(unused)]
use inline_tweak::tweak;

#[derive(Clone, PartialEq)]
pub struct Blip<B: Brain> {
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
        olds: &[Self],
        tree: &RTree<BlipLoc>,
        foodgrid: &FoodGrid,
        time: f64,
        dt: f64,
    ) -> Option<Self>
    where
        B: Copy,
    {
        let mut inputs: brains::Inputs = Default::default();

        let search = tree
            .locate_within_distance(old.status.pos, LOCAL_ENV)
            .filter(|d| d.position() != &old.status.pos);

        for nb in search {
            // sound
            let nb = &olds[nb.data];
            use rstar::PointDistance;
            let dist_squared = PointDistance::distance_2(&old.status.pos, &nb.status.pos);

            // todo: get rid of sqrt
            let nb_sound = (nb.status.vel[0] * nb.status.vel[0])
                + (nb.status.vel[1] * nb.status.vel[1]).sqrt();

            *inputs.sound_mut() += nb_sound / dist_squared;
        }

        // rust apparently does the modulo [-pi/2, pi/2] internally
        *inputs.clock1_mut() = (time * old.genes.clockstretch_1).sin();
        *inputs.clock2_mut() = (time * old.genes.clockstretch_2).sin();

        let gridpos = [
            (self.status.pos[0] / 10.) as usize,
            (self.status.pos[1] / 10.) as usize,
        ];
        let grid_slot = &foodgrid[gridpos[0]][gridpos[1]];

        let casstat = Atomic::new(0usize);

        // there is no synchronization between threads, only the global food object
        // so only the atomic operaton on it needs to be taken care of.
        // there is no other operations to synchronize
        // relaxed ordering should therefore be fine to my best knowledge
        let mut grid_value = grid_slot.load(Ordering::Relaxed);
        let mut outputs;
        loop {
            *inputs.smell_mut() = grid_value;

            // inputs are processed at this point, time to feed the brain some data
            outputs = old.genes.brain.think(&inputs);

            // eat food
            //todo: consider using actual speed, not acceleration
            //todo also this needs to be finetuned quite a bit in paralel with food spawning and
            //movement speed, rn they are very very slow in general and just zoom over food,
            //fully consuming it
            let speed = outputs.speed() * tweak!(1.);
            // eat at most half the field/second
            let eating = (grid_value / speed.max(1.)) * dt;
            if eating > 0. && !eating.is_nan() {
                let newval = grid_value - eating;
                match grid_slot.compare_exchange(
                    grid_value,
                    newval,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        self.status.food += eating;
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
            println!("[{}]: had to cas-loop {} times", time, casstat);
        }
        let digest = self.status.food.min(4. * dt);
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

        let exhaustion = (0.1 + (outputs.speed().abs() * 0.4)) * dt * 0.1;
        self.status.hp -= exhaustion;

        // reproduce & mutate
        // reproduction is a bit of a problem since it needs to ad new entries to the vec
        // which is kinda bad for multithreading.
        // its a rather rare event though so its special-cased

        let ret = if self.status.hp > old.genes.repr_tres {
            println!("new spawn!");
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

        pos[0] %= SIM_WIDTH;
        pos[1] %= SIM_HEIGHT;

        if pos[0] < 0. {
            pos[0] += SIM_WIDTH
        };
        if pos[1] < 0. {
            pos[1] += SIM_WIDTH
        };
    }
    pub fn new<R: Rng>(mut rng: R) -> Self {
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
                rgb: [0.; 4],
            },
            genes: Genes::new(&mut rng),
        }
    }
    pub fn split<R: Rng>(&mut self, mut rng: R) -> Self
    where
        B: Clone + Copy,
    {
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

#[derive(Copy, Clone, PartialEq)]
pub struct Genes<B: Brain> {
    pub brain: B,
    pub mutation_rate: f64,
    pub repr_tres: f64,
    // actual clock is multiplied by this
    pub clockstretch_1: f64,
    pub clockstretch_2: f64,
    // 3 eyes, each represented by an angle in radians [-pi-pi]
    pub eyes: [f64; 3],
}

impl<B: Brain> Genes<B> {
    fn new<R: Rng>(mut rng: R) -> Self {
        use std::f64::consts::PI;
        Self {
            brain: B::init(&mut rng),
            mutation_rate: rng.gen_range(-0.001, 0.001) + 0.01,
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

    fn mutate<R: Rng>(&self, mut rng: R) -> Self
    where
        B: Copy,
    {
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

pub fn eyefilter<'a, I, T, F>(
    env: I,
    status: &'a Status,
    eye: f64,
    fov: f64,
    map: F,
) -> impl Iterator<Item = T> + 'a
where
    I: Iterator<Item = T> + 'a,
    F: Fn(&T) -> Vector + 'a,
{
    let base_angle = vecmath::atan2(vecmath::norm(status.vel));
    let eye_angle = vecmath::rad_norm(base_angle + eye);
    env.filter(move |t| {
        let pos = map(t);
        let diff = vecmath::sub(pos, status.pos);
        let diff_norm = vecmath::norm(diff);
        let angle = vecmath::atan2(diff_norm);
        let angle_diff = vecmath::rad_norm(angle - eye_angle);
        angle_diff.abs() < fov
    })
}
