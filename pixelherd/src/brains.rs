#![allow(unused)]
use crate::blip::scaled_rand;
use crate::config;
use rand::Rng;
use std::convert::TryInto;

use opengl_graphics::GlGraphics;
// 3 memory cells
// eyes are 3 colours, angle, dist
const N_INPUTS: usize = 5 + 3 + (config::b::N_EYES * 5);
const N_OUTPUTS: usize = 6 + 3;

const INNER_SIZE: usize = (N_INPUTS + N_OUTPUTS) / 2;

/// stored as an array for easy
/// neural network access.
/// but accessed/modified through methods
//todo: scale all inputs to ~-10-10
#[derive(Clone, PartialEq, Default, Debug)]
pub struct Inputs {
    data: [f64; N_INPUTS],
}

impl Inputs {
    pub fn sound_mut(&mut self) -> &mut f64 {
        &mut self.data[0]
    }
    pub fn smell_mut(&mut self) -> &mut f64 {
        &mut self.data[1]
    }
    pub fn smell_dist_mut(&mut self) -> &mut f64 {
        &mut self.data[2]
    }
    pub fn clock1_mut(&mut self) -> &mut f64 {
        &mut self.data[3]
    }
    pub fn clock2_mut(&mut self) -> &mut f64 {
        &mut self.data[4]
    }
    pub fn memory_mut(&mut self) -> &mut [f64; 3] {
        use std::convert::TryInto;
        (&mut self.data[5..(5 + 3)]).try_into().unwrap()
    }
    pub fn eyes_mut(&mut self) -> [&mut [f64]; config::b::N_EYES] {
        let data = &mut self.data[(5 + 3)..];
        let eyesize = 5;
        // sadly arrays and iterators don't interact well currently
        // so this is more or less hardcoded for now
        let (one, data) = data.split_at_mut(eyesize);
        let (two, data) = data.split_at_mut(eyesize);
        let (three, data) = data.split_at_mut(eyesize);
        [one, two, three]
    }
}

/// stored as an array for easy
/// neural network access.
/// but accessed/modified through methods
#[derive(Clone, PartialEq, Default, Debug)]
pub struct Outputs {
    data: [f64; N_OUTPUTS],
}
impl Outputs {
    // fixme: shouldn't this be 0-1?
    /// (0, 1)
    pub fn spike(&self) -> f64 {
        self.data[0] + 0.5
    }
    /// (-0.5, 0.5)
    pub fn steering(&self) -> f64 {
        self.data[1]
    }
    /// (0, MAX_SPEED)
    pub fn speed(&self) -> f64 {
        // (-0.5, 0.5)
        let mut speed = self.data[2];
        // (-0.05, 0.5)
        if speed <= 0. {
            speed /= 10.
        }
        // (0, 0.55)
        speed += 0.05;
        // (0, 1)
        speed /= 0.55;
        speed *= config::b::MAX_SPEED;
        speed
    }
    // these are [-0.5, 0.5]
    pub fn rgb_raw(&self) -> &[f64; 3] {
        self.data[3..=5].try_into().unwrap()
    }
    pub fn r(&self) -> f64 {
        self.data[3] + 0.5
    }
    pub fn b(&self) -> f64 {
        self.data[4] + 0.5
    }
    pub fn g(&self) -> f64 {
        self.data[5] + 0.5
    }
    pub fn memory(&self) -> &[f64; 3] {
        self.data[6..(6 + 3)].try_into().unwrap()
    }
}
#[test]
fn rgbsize() {
    let o = Outputs::default();
    let rgb = o.rgb_raw();
}

pub trait Brain: Clone {
    fn init<R: Rng>(rng: R) -> Self;
    fn think(&self, inputs: &Inputs) -> Outputs;
    fn mutate<R: Rng>(&mut self, rng: R, rate: f64);
    fn draw(&self, gl: &mut GlGraphics, trans: [[f64; 3]; 2]);
}

// todo: bigbrain is currently too inflexible, it just outputs values very close to 0 for basically
// any input. maybe i can do a 2-step-process where i first use a 1-layer brain
// and then add another layer between it and the input (or the output) with most weights
// initialized to ~0, and one to ~1.
#[derive(Copy, Clone, Default, PartialEq, Debug, Serialize, Deserialize)]
pub struct BigBrain {
    // each output gets a weight for each input
    in2mid: [[f64; N_INPUTS]; INNER_SIZE],
    mid_bias: [f64; INNER_SIZE],
    mid2out: [[f64; INNER_SIZE]; N_OUTPUTS],
    out_bias: [f64; N_OUTPUTS],
    // todo: maybe i can add some loopback values
}

impl Brain for BigBrain {
    fn draw(&self, gl: &mut GlGraphics, trans: [[f64; 3]; 2]) {
        todo!()
    }
    fn mutate<R: Rng>(&mut self, mut rng: R, rate: f64) {
        for mid in &mut self.in2mid {
            for inp in mid.iter_mut() {
                scaled_rand(&mut rng, rate, 0.1, 0.1, inp);
            }
        }

        for out in &mut self.mid2out {
            for mid in out.iter_mut() {
                scaled_rand(&mut rng, rate, 0.1, 0.1, mid);
            }
        }

        for bias in self.mid_bias.iter_mut() {
            scaled_rand(&mut rng, rate, 0.01, 0.01, bias);
        }

        for bias in self.out_bias.iter_mut() {
            scaled_rand(&mut rng, rate, 0.01, 0.01, bias);
        }
    }
    fn init<R: Rng>(mut r: R) -> Self {
        let mut s: Self = Default::default();
        for mid in &mut s.in2mid {
            for inp in mid.iter_mut() {
                *inp = r.random_range(-0.1..0.1);
            }
        }
        for out in &mut s.mid2out {
            for mid in out.iter_mut() {
                *mid = r.random_range(-0.1..0.1);
            }
        }
        for bias in &mut s.mid_bias {
            *bias = r.random_range(-0.01..0.01);
        }
        for bias in &mut s.out_bias {
            *bias = r.random_range(-0.01..0.01);
        }
        s
    }
    // todo: maybe use my own lib for this
    // no direct learing is happening so maybe not
    fn think(&self, inputs: &Inputs) -> Outputs {
        let mut mid = [0.0_f64; INNER_SIZE];

        for ((iw, m), bias) in self
            .in2mid
            .iter()
            .zip(mid.iter_mut())
            .zip(self.mid_bias.iter())
        {
            assert_eq!(iw.len(), inputs.data.len());
            let weighted_in: f64 = iw.iter().zip(&inputs.data).map(|(iw, i)| iw * i).sum();
            let weighted_in = weighted_in + bias;
            let clamped = weighted_in.max(-20.).min(20.);
            let res = 1. / (1. + (-clamped).exp());
            // center sigmoid around 0
            *m = res - 0.5;
        }

        let mut o: Outputs = Outputs::default();
        for ((mw, o), bias) in self
            .mid2out
            .iter()
            .zip(o.data.iter_mut())
            .zip(self.out_bias.iter())
        {
            let weighted_in: f64 = mw.iter().zip(&mid).map(|(mw, m)| mw * m).sum();
            let weighted_in = weighted_in + bias;
            let clamped = weighted_in.max(-20.).min(20.);
            let res = 1. / (1. + (-clamped).exp());
            // center sigmoid around 0
            *o = res - 0.5;
        }
        o
    }
}

use serde_derive::{Deserialize, Serialize};

#[derive(Copy, Clone, Default, PartialEq, Debug, Serialize, Deserialize)]
pub struct SimpleBrain {
    // each output gets a weight for each input
    weights: [[f64; N_INPUTS]; INNER_SIZE],
    bias: [f64; INNER_SIZE],
}

impl Brain for SimpleBrain {
    fn draw(&self, gl: &mut GlGraphics, trans: [[f64; 3]; 2]) {
        use graphics::Transformed;
        for (out_n, out_v) in self.weights.iter().enumerate() {
            for (in_n, in_v) in out_v.iter().enumerate() {
                let base = -(INNER_SIZE as f64);
                let out_n = out_n as f64;
                // draw from left to right, but starting far enough to the left so it ends right
                // at the centre line to avoid overlap
                let h = (base + out_n) * 10.;
                let v = (in_n * 10) as f64;
                let transform = trans.trans(h, v);

                let pos = if *in_v > 0. { 1. } else { 0. };
                let neg = 1. - pos;
                let col = [pos, 0., neg, (in_v.abs() + 0.5) as f32];

                graphics::rectangle([1.; 4], [0., 0., 10., 10.], transform, gl);
                graphics::rectangle(col, [0., 0., 10., 10.], transform, gl)
            }
        }
    }
    fn mutate<R: Rng>(&mut self, mut rng: R, rate: f64) {
        for out in &mut self.weights {
            for inp in out.iter_mut() {
                if rng.random_range(0..100) < 5 {
                    scaled_rand(&mut rng, rate, 0.2, 0.2, inp);
                }
            }
        }

        for bias in self.bias.iter_mut() {
            if rng.random_range(0..100) < 5 {
                scaled_rand(&mut rng, rate, 0.2, 0.2, bias);
            }
        }
    }
    fn init<R: Rng>(mut r: R) -> Self {
        let mut s: Self = Default::default();
        for out in &mut s.weights {
            for inp in out.iter_mut() {
                *inp = r.random_range(-0.1..0.1);
            }
        }
        for bias in &mut s.bias {
            *bias = r.random_range(-0.01..0.01);
        }
        s
    }
    // todo: maybe use my own lib for this
    // no direct learing is happening so maybe not
    fn think(&self, inputs: &Inputs) -> Outputs {
        let mut o: Outputs = Outputs::default();
        for ((iw, o), bias) in self
            .weights
            .iter()
            .zip(o.data.iter_mut())
            .zip(self.bias.iter())
        {
            let weighted_in: f64 = iw.iter().zip(&inputs.data).map(|(iw, i)| iw * i).sum();
            let weighted_in = weighted_in + bias;
            let clamped = weighted_in.max(-20.).min(20.);
            let res = 1. / (1. + (-clamped).exp());
            // center sigmoid around 0
            *o = res - 0.5;
        }
        o
    }
}

impl<B: Brain> Brain for Box<B> {
    fn draw(&self, gl: &mut GlGraphics, trans: [[f64; 3]; 2]) {
        self.as_ref().draw(gl, trans)
    }
    fn mutate<R: Rng>(&mut self, rng: R, rate: f64) {
        self.as_mut().mutate(rng, rate)
    }
    fn init<R: Rng>(r: R) -> Self {
        Box::new(B::init(r))
    }
    fn think(&self, inputs: &Inputs) -> Outputs {
        self.as_ref().think(inputs)
    }
}

#[test]
fn arrayacc() {
    let arr = [0, 1, 2, 3, 4, 5, 6, 7];
    assert_eq!(arr[3], 3);
    assert_eq!(arr[4..][0], 4);
}
