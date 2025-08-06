use rand::{ distr::Distribution };
use rand_distr::Normal;

use crate::{ rotta_rs_module::Module };

impl Module {
    pub fn glorot_initialization(&mut self, input: usize, output: usize) -> f32 {
        let glorot_formula = 2.0 / ((input + output) as f32);
        let glorot_std = glorot_formula.sqrt();

        let normal = Normal::new(0.0, glorot_std).unwrap();
        normal.sample(&mut self.rng)
    }
}
