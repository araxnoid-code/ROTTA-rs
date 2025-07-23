mod gru;
pub use gru::*;

pub use rand::distr::Uniform;
use rand_distr::Distribution;

use crate::{ Module, Tensor };

impl Module {
    pub fn gru_init(&mut self, hidden: usize) -> Gru {
        let uniform = Uniform::new(
            -(1.0 / (hidden as f64)).sqrt(),
            (1.0 / (hidden as f64)).sqrt()
        ).unwrap();

        // reset parameters
        let w_r = Tensor::from_shape_fn(vec![hidden * 2, hidden], || {
            uniform.sample(&mut self.rng)
        });
        self.parameters.lock().unwrap().push(w_r.node.clone());

        let b_r = Tensor::from_shape_fn(vec![1, hidden], || { uniform.sample(&mut self.rng) });
        self.parameters.lock().unwrap().push(b_r.node.clone());

        // update parameters
        let w_u = Tensor::from_shape_fn(vec![hidden * 2, hidden], || {
            uniform.sample(&mut self.rng)
        });
        self.parameters.lock().unwrap().push(w_u.node.clone());

        let b_u = Tensor::from_shape_fn(vec![1, hidden], || { uniform.sample(&mut self.rng) });
        self.parameters.lock().unwrap().push(b_u.node.clone());

        // candidate parameters
        let w_c = Tensor::from_shape_fn(vec![hidden * 2, hidden], || {
            uniform.sample(&mut self.rng)
        });
        self.parameters.lock().unwrap().push(w_c.node.clone());

        let b_c = Tensor::from_shape_fn(vec![1, hidden], || { uniform.sample(&mut self.rng) });
        self.parameters.lock().unwrap().push(b_c.node.clone());

        Gru {
            b_c,
            b_r,
            b_u,
            w_c,
            w_r,
            w_u,
        }
    }
}
