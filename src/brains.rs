#![allow(unused)]
use crate::{scaled_rand, Inputs, Outputs};
use crate::{N_INPUTS, N_OUTPUTS};
use rand::Rng;
const INNER_SIZE: usize = (N_INPUTS + N_OUTPUTS) / 2;

#[derive(Copy, Clone, Default, PartialEq)]
pub struct BigBrain {
    // each output gets a weight for each input
    in2mid: [[f64; N_INPUTS]; INNER_SIZE],
    mid_bias: [f64; INNER_SIZE],
    mid2out: [[f64; INNER_SIZE]; N_OUTPUTS],
    out_bias: [f64; N_OUTPUTS],
}

impl BigBrain {
    pub fn mutate<R: Rng>(&mut self, mut rng: R, rate: f64) {
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
}

impl BigBrain {
    pub fn init<R: Rng>(mut r: R) -> Self {
        let mut s: Self = Default::default();
        for mid in &mut s.in2mid {
            for inp in mid.iter_mut() {
                *inp = r.gen_range(-0.1, 0.1);
            }
        }
        for out in &mut s.mid2out {
            for mid in out.iter_mut() {
                *mid = r.gen_range(-0.1, 0.1);
            }
        }
        for bias in &mut s.mid_bias {
            *bias = r.gen_range(-0.01, 0.01);
        }
        for bias in &mut s.out_bias {
            *bias = r.gen_range(-0.01, 0.01);
        }
        s
    }
    // todo: maybe use my own lib for this
    // no direct learing is happening so maybe not
    pub(crate) fn think(&self, inputs: &Inputs) -> Outputs {
        let mut mid = [0.0_f64; INNER_SIZE];

        for ((iw, m), bias) in self
            .in2mid
            .iter()
            .zip(mid.iter_mut())
            .zip(self.mid_bias.iter())
        {
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

#[derive(Copy, Clone, Default, PartialEq)]
pub struct SimpleBrain {
    // each output gets a weight for each input
    weights: [[f64; N_INPUTS]; INNER_SIZE],
    bias: [f64; INNER_SIZE],
}

impl SimpleBrain {
    pub fn mutate<R: Rng>(&mut self, mut rng: R, rate: f64) {
        for out in &mut self.weights {
            for inp in out.iter_mut() {
                scaled_rand(&mut rng, rate, 0.1, 0.1, inp);
            }
        }

        for bias in self.bias.iter_mut() {
            scaled_rand(&mut rng, rate, 0.01, 0.01, bias);
        }
    }
}

impl SimpleBrain {
    pub fn init<R: Rng>(mut r: R) -> Self {
        let mut s: Self = Default::default();
        for out in &mut s.weights {
            for inp in out.iter_mut() {
                *inp = r.gen_range(-0.1, 0.1);
            }
        }
        for bias in &mut s.bias {
            *bias = r.gen_range(-0.01, 0.01);
        }
        s
    }
    // todo: maybe use my own lib for this
    // no direct learing is happening so maybe not
    pub(crate) fn think(&self, inputs: &Inputs) -> Outputs {
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
