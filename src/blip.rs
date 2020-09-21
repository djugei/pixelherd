use crate::SIM_HEIGHT;
use crate::SIM_WIDTH;
use rand::Rng;

use crate::brains::Brain;

#[derive(Clone, PartialEq)]
pub struct Blip<B: Brain> {
    /// things that change during the lifetime of a blip
    pub status: Status,
    /// things that only change trough mutation during reproduction
    pub genes: Genes<B>,
}

impl<B: Brain> Blip<B> {
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
}

#[derive(Copy, Clone, PartialEq)]
pub struct Genes<B: Brain> {
    pub brain: B,
    pub mutation_rate: f64,
    pub repr_tres: f64,
    // actual clock is multiplied by this
    pub clockstretch_1: f64,
    pub clockstretch_2: f64,
}

impl<B: Brain> Genes<B> {
    fn new<R: Rng>(mut rng: R) -> Self {
        Self {
            brain: B::init(&mut rng),
            mutation_rate: rng.gen_range(-0.001, 0.001) + 0.01,
            repr_tres: rng.gen_range(-10., 10.) + 100.,
            clockstretch_1: rng.gen_range(0.01, 1.),
            clockstretch_2: rng.gen_range(0.01, 1.),
        }
    }
    fn mutate<R: Rng>(&self, mut rng: R) -> Self
    where
        B: Copy,
    {
        let mut new = self.clone();
        new.brain.mutate(&mut rng, self.mutation_rate);
        new.repr_tres *= 1. + rng.gen_range(-self.mutation_rate, self.mutation_rate);
        new.mutation_rate += rng.gen_range(-self.mutation_rate, self.mutation_rate) / 10.;

        new.clockstretch_1 *= 1. + rng.gen_range(-self.mutation_rate, self.mutation_rate);
        new.clockstretch_2 *= 1. + rng.gen_range(-self.mutation_rate, self.mutation_rate);
        new
    }
}
