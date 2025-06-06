use rand::{ distr::{ self, Distribution }, rng, rngs::{ self, ThreadRng }, Rng, SeedableRng };
use rand_distr::Normal;

use crate::rotta_rs::Module;

impl Module {
    pub fn glorot_initialization(&mut self, input: usize, output: usize) -> f64 {
        let glorot_formula = 2.0 / ((input + output) as f64);
        let glorot_std = glorot_formula.sqrt();

        let normal = Normal::new(0.0, glorot_std).unwrap();
        normal.sample(&mut self.rng)
    }
}
