use rand_distr::{ Distribution, Normal };

use crate::rotta_rs::Module;

impl Module {
    pub fn he_initialization(&mut self, input: usize) -> f64 {
        let he = 2.0 / (input as f64);
        let he_std = he.sqrt();

        let normal = Normal::new(0.0, he_std).unwrap();
        normal.sample(&mut self.rng)
    }
}
