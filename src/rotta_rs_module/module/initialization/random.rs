use rand::Rng;

use crate::rotta_rs_module::Module;

impl Module {
    pub fn random_initialization(&mut self) -> f32 {
        let random = self.rng.random();
        random
    }
}
