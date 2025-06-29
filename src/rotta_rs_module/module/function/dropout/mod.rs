mod dropout;
use std::sync::{ Arc, Mutex };

pub use dropout::*;
use crate::rotta_rs_module::Module;
use rand::distr::Bernoulli;

impl Module {
    pub fn dropout_init(&mut self, p: f64) -> Dropout {
        let bernouli = Bernoulli::new(p).unwrap();

        let train_status = Arc::new(Mutex::new(false));
        self.eval_handlers.push(train_status.clone());
        Dropout {
            bernoulli: bernouli,
            p,
            rng: self.rng.clone(),
            eval: train_status.clone(),
        }
    }
}
