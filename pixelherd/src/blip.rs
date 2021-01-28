use crate::config;
use rand::Rng;

use crate::brains;
use crate::brains::Brain;

use crate::app::TreeRef;
use crate::app::{MeatGrid, OldMeatGrid, OldVegGrid, VegGrid};

use crate::vecmath;
use crate::vecmath::Vector;

use atomic::Ordering;

#[derive(PartialEq, Debug)]
pub struct Blip<'s, 'g, B: Brain> {
    /// things that change during the lifetime of a blip
    pub status: &'s mut Status,
    /// things that only change trough mutation during reproduction
    pub genes: &'g Genes<B>,
}

impl<'s, 'g, B: Brain> Blip<'s, 'g, B> {
    // consider changing the signature to
    // &old_status, &genes -> (new_status, option<newblip>)
    // which could avoid a memcopy by MaybeUninit-initializing the new array
    // and filling it with the new blips
    pub fn update<R: Rng>(
        &mut self,
        mut rng: R,
        olds: &[Status],
        oldg: &[Genes<B>],
        tree: TreeRef,
        oldveg: &OldVegGrid,
        veg: &VegGrid,
        oldmeat: &OldMeatGrid,
        meat: &MeatGrid,
        time: f64,
        dt: f64,
    ) -> Option<(Status, Genes<B>)> {
        // todo: split into input gathering, thinking and etc
        let mut inputs: brains::Inputs = Default::default();
        *inputs.memory_mut() = self.status.memory;

        let search = tree.query_distance(&self.status.pos, config::b::LOCAL_ENV);

        let own_base_angle = base_angle(&self.status);
        let mut eyedists = [(f64::INFINITY, 0., 0); config::b::N_EYES];
        let mut eye_angles = [0.; config::b::N_EYES];
        for (angle, eye) in eye_angles.iter_mut().zip(&self.genes.eyes) {
            *angle = eye_angle(own_base_angle, *eye)
        }

        let mut spiked = false;

        for (dist_squared, (_p, index)) in search {
            // sound
            let nb = &olds.get(*index).unwrap();

            // todo: get rid of sqrt
            let nb_sound = (nb.vel[0] * nb.vel[0]) + (nb.vel[1] * nb.vel[1]).sqrt();

            *inputs.sound_mut() += nb_sound / dist_squared;

            for (i, eye) in eye_angles.iter().enumerate() {
                // 1/10th of a full circle
                // todo: evolve this
                let fov = 0.1 * std::f64::consts::TAU;
                let angle = eye_vision(&self.status, *eye, nb.pos);
                // see the closest one in fov
                // todo: i can sort the searches by distance
                //  + faster angle calcuations as i only need to get the first matching (atan2 is
                //    expensive)
                //  - need to actually alloc&sort the search
                if angle.abs() < fov && eyedists[i].0 > dist_squared {
                    eyedists[i] = (dist_squared, angle, *index);
                }
            }
            for (&(dis, angle, id), inp) in eyedists.iter().zip(inputs.eyes_mut().iter_mut()) {
                if dis != f64::INFINITY {
                    let nb = &olds.get(id).unwrap();
                    let rgb = nb.rgb;
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
            // todo: since this is doing a second range check it could be good to do a separate
            // query, but i am not sure if this would actually speed things up.
            if dist_squared < (5. * 5.) {
                // don't spike others on each hit, only once
                if self.status.spike > 0.3 {
                    // todo: rename eye_vision, spike is an "eye" pointing straight ahead.
                    let col_angle = eye_vision(&self.status, own_base_angle, nb.pos).abs();
                    if col_angle < (0.05 * std::f64::consts::TAU) {
                        self.status.spike -= 0.3 * dt;
                    }
                }
                // get spiked
                if nb.spike > 0.3 {
                    let nbg = &oldg.get(*index).unwrap();
                    if nbg.vore > 0.3 {
                        let relspeed = vecmath::len(vecmath::add(self.status.vel, nb.vel));
                        if relspeed > 1.5 {
                            let other_base = base_angle(&nb);
                            let col_angle = eye_vision(&nb, other_base, self.status.pos).abs();
                            // in range, they have an extended spike, pointing at us, significant
                            // speed difference, they are not fully herbivore
                            if col_angle < (0.05 * std::f64::consts::TAU) {
                                // i should really move to fixed step size.
                                let damageconst = 1.;
                                let damage = damageconst * relspeed * nbg.vore * nb.spike * dt;
                                //println!("took {} damage from {}", damage, index);
                                self.status.hp -= damage;
                                spiked = true;
                            }
                        }
                    }
                }
            }
        }

        // rust apparently does the modulo [-pi/2, pi/2] internally
        *inputs.clock1_mut() = (time * self.genes.clockstretch_1).sin();
        *inputs.clock2_mut() = (time * self.genes.clockstretch_2).sin();

        let x = self.status.pos[0] / 10.;
        let y = self.status.pos[1] / 10.;
        let gridpos = [x as usize, y as usize];
        let grid_slot = &veg[gridpos[0]][gridpos[1]];

        let grid_value_r = oldveg[gridpos[0]][gridpos[1]];
        let grid_value = grid_value_r.to_f64();

        let xfrac = x.fract();
        let yfrac = y.fract();
        let centerdist = xfrac * xfrac + yfrac * yfrac;
        *inputs.smell_dist_mut() = centerdist / 2.;

        // inputs are processed at this point, time to feed the brain some data
        let outputs = self.genes.brain.think(&inputs);

        // eat food
        //todo: put into config
        let consumption_c = |grid_value: f64| {
            // can eat 10 food / second (arbitrary)
            let max = 10. * dt;
            // half consumption speed on basically empty square
            // full consumption on 5 food, double on 15
            let gridfactor = 0.5 + (grid_value / 10.);
            // 1..11
            let div = 1. + (outputs.speed() * 2.5);
            (max * gridfactor / div).min(grid_value)
        };
        let consumption = (consumption_c)(grid_value) * (1. - self.genes.vore);
        if consumption > 0. && !consumption.is_nan() {
            let consumption_r: fix_rat::TenRat = consumption.into();
            // there is no synchronization between threads, only the global food object
            // so only the atomic operaton on it needs to be taken care of.
            // there is no other operations to synchronize
            // relaxed ordering should therefore be fine to my best knowledge
            // gotta do all deterministic calculations in rationals
            grid_slot
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                    Some(old - consumption_r)
                })
                .unwrap();
            self.status.food += consumption;
        }

        let meat_grid_slot = &meat[gridpos[0]][gridpos[1]];

        let meat_grid_value_r = oldmeat[gridpos[0]][gridpos[1]];
        let meat_grid_value = meat_grid_value_r.to_f64();
        let meat_consumption = if self.genes.vore < 0.2 {
            0.
        } else {
            // can eat 10 food / second (arbitrary)
            let max = 10. * dt;
            // half consumption speed on basically empty square
            // full consumption on 5 food, double on 15
            let gridfactor = 0.5 + (meat_grid_value / 10.);
            // meat consumption is not limited by speed
            let consumption = (max * gridfactor).min(meat_grid_value).min(0.1);
            consumption * self.genes.vore * 2.
        };
        if meat_consumption > 0. && !meat_consumption.is_nan() {
            let consumption_r: fix_rat::HundRat = consumption.into();
            // never fails cause we never return None
            meat_grid_slot
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                    let new = old - consumption_r;
                    Some(new)
                })
                .unwrap();
            self.status.food += consumption;
        }
        // food is ~ 0-10+, scale to -5 to 5+
        // maybe split this into two inputs
        *inputs.smell_mut() = ((grid_value - 5.) * (1. - self.genes.vore))
            + ((meat_grid_value - 5.) * (self.genes.vore));

        self.status
            .memory
            .iter_mut()
            .zip(outputs.memory())
            // slowly change memory to output
            // todo: maybe do this different for each memory field (short/long memory)
            .for_each(|(m, nm)| *m = (*m * (1. - dt)) + (nm * dt));

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
        let spike = self.status.spike * (1. - dt);
        let spike = spike + (outputs.spike() * dt);
        self.status.spike = spike.min(1.).max(0.);

        // change direction
        let steer = [0., outputs.steering()];
        // fixme: handle 0 speed
        let dir = vecmath::norm(self.status.vel);

        let push = vecmath::rotate(steer, dir);

        let mut vel = vecmath::add(self.status.vel, push);
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
        // reproduction is a bit of a problem since it needs to add new entries to the vec
        // which is kinda bad for multithreading.
        // its a rather rare event though so its special-cased

        let ret = if self.status.hp > self.genes.repr_tres {
            let spawn = self.split(&mut rng);
            Some(spawn)
        } else {
            None
        };
        self.status.age += dt;

        // time to die :)
        if self.status.hp < 0. {
            let age = self.status.age;
            // todo: make this configurable
            // todo: don't directly use age, use minimum consumed food during age
            // first bit of food is directly given
            let food = age.min(50.);
            let remain = age - food;
            if spiked {
                println!("dying from spiking")
            };
            // give less food if died of starvation
            let spiked = if spiked { 1. } else { 0.5 };
            // remainder is log2-ed
            let total_food = (food + (remain + 1.).log2()) * spiked;

            // more food in center, less spilled
            let rads = [0, 1, 2];
            let foods = [total_food / 2., total_food / 4., total_food / 4.];
            for (rad, food) in rads.iter().zip(&foods) {
                let diam = (2 * rad) + 1;
                let area = diam * diam;
                let food = food / (area as f64);
                let food = fix_rat::HundRat::aprox_float_fast(food).unwrap();
                let range = crate::app::foodenv(gridpos, *rad);

                for pos in range {
                    let slot = &meat[pos[0]][pos[1]];
                    slot.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                        // this is not a great way to handle this. ideas/observations:
                        // 2) hand off to synchronous code by setting hp to -inf or smth (need to
                        //    roll back previous changes though!)
                        // 3) can't handle this in parallel code cause it would break determinism in
                        //    every case.
                        // 4) this is insanely rare
                        // todo: meat is now ~0-100 adjust assumptions accordingly
                        old.checked_add(food)
                    })
                    .unwrap();
                }
            }
        }
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
    pub fn new<R: Rng>(mut rng: R) -> (Status, Genes<B>) {
        let x = rng.gen_range(0.0..config::SIM_WIDTH);
        let y = rng.gen_range(0.0..config::SIM_HEIGHT);

        let dx = rng.gen_range(-30.0..30.);
        let dy = rng.gen_range(-5.0..5.);
        (
            Status {
                pos: [x, y],
                vel: [dx, dy],
                spike: 0.,
                hp: 25.,
                food: 5.,
                age: 0.,
                children: 0,
                generation: 0,
                rgb: [0.; 4],
                memory: [0.; 3],
                lineage: 0.,
            },
            Genes::new(&mut rng),
        )
    }
    pub fn from_components(status: &'s mut Status, genes: &'g Genes<B>) -> Self {
        Self { status, genes }
    }
    pub fn split<R: Rng>(&mut self, mut rng: R) -> (Status, Genes<B>) {
        self.status.hp /= 2.;
        self.status.children += 1;
        let mut new_status = *self.status;
        let new_genes = self.genes.mutate(&mut rng);
        new_status.generation += 1;
        new_status.food = 0.;
        new_status.age = 0.;
        new_status.children = 0;
        new_status.lineage += self.status.age;
        // push child away a bit
        new_status.vel[0] += 1.;
        new_status.vel[1] += 1.;

        (new_status, new_genes)
    }
}

use serde_derive::{Deserialize, Serialize};

#[derive(Copy, Clone, PartialEq, Default, Debug, Serialize, Deserialize)]
pub struct Status {
    pub pos: [f64; 2],
    pub vel: [f64; 2],
    pub spike: f64,
    pub food: f64,
    pub hp: f64,
    // instead of age store birth time maybe
    pub age: f64,
    pub children: usize,
    pub generation: usize,
    pub rgb: [f32; 4],
    pub memory: [f64; 3],
    // the total age of ancestors until my birth
    // this is kindof more of a gene cause it never changes
    pub lineage: f64,
}

#[derive(Copy, Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Genes<B: Brain> {
    pub brain: B,
    pub mutation_rate: f64,
    pub repr_tres: f64,
    // actual clock is multiplied by this
    pub clockstretch_1: f64,
    pub clockstretch_2: f64,
    // 3 eyes, each represented by an angle in radians [-pi-pi]
    pub eyes: [f64; config::b::N_EYES],
    // 0-1 0: fully herbivore, 1 fully carnivore
    pub vore: f64,
}

impl<B: Brain> Genes<B> {
    fn new<R: Rng>(mut rng: R) -> Self {
        use std::f64::consts::PI;
        Self {
            brain: B::init(&mut rng),
            mutation_rate: (rng.gen_range(-0.001..0.001) + 0.01) * 4.,
            repr_tres: rng.gen_range(-10.0..10.) + 100.,
            clockstretch_1: rng.gen_range(0.01..1.),
            clockstretch_2: rng.gen_range(0.01..1.),
            eyes: [
                rng.gen_range(-PI..PI),
                rng.gen_range(-PI..PI),
                rng.gen_range(-PI..PI),
            ],
            //eyes: [(-2. / 3.) * PI, 0., (2. / 3.) * PI],
            //eyes: [0., 0., 0.],
            vore: rng.gen_range(0.0..1.),
        }
    }

    // lol, ok figured out why evolution was not happening
    #[must_use]
    fn mutate<R: Rng>(&self, mut rng: R) -> Self {
        //let mut new = self.clone();
        let mut brain = self.brain.clone();
        brain.mutate(&mut rng, self.mutation_rate);

        let repr_tres =
            self.repr_tres * (1. + rng.gen_range(-self.mutation_rate..self.mutation_rate));

        let mutation_rate =
            self.mutation_rate + (rng.gen_range(-self.mutation_rate..self.mutation_rate) / 10.);

        let clockstretch_1 =
            self.clockstretch_1 * (1. + rng.gen_range(-self.mutation_rate..self.mutation_rate));
        let clockstretch_2 =
            self.clockstretch_2 * (1. + rng.gen_range(-self.mutation_rate..self.mutation_rate));

        use std::f64::consts::PI;
        let mut eyes = self.eyes;
        for eye in eyes.iter_mut() {
            *eye += rng.gen_range((-self.mutation_rate * PI)..(self.mutation_rate * PI));
            // wrap around
            *eye %= PI;
        }

        let vore = self.vore
            + rng
                .gen_range(-self.mutation_rate..self.mutation_rate)
                .max(0.)
                .min(1.);
        Self {
            brain,
            repr_tres,
            mutation_rate,
            clockstretch_1,
            clockstretch_2,
            eyes,
            vore,
        }
    }
}
pub fn scaled_rand<R: Rng>(mut r: R, rate: f64, abs_scale: f64, mul_scale: f64, val: &mut f64) {
    *val += r.gen_range(-abs_scale..abs_scale) * r.gen_range(0.0..rate);
    *val *= 1. + (r.gen_range(-mul_scale..mul_scale) * r.gen_range(0.0..rate));
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
    // absolute heading of the eye
    eye_angle: f64,
    other: Vector,
) -> f64 {
    let diff = vecmath::sub(other, status.pos);
    let diff_norm = vecmath::norm(diff);
    let angle = vecmath::atan2(diff_norm);
    vecmath::rad_norm(angle - eye_angle)
}
