mod lstm;
pub use lstm::*;
pub use rand::distr::Uniform;
use rand_distr::Distribution;

use crate::{ Module, Tensor };

impl Module {
    pub fn lstm_init(&mut self, hidden: usize) -> Lstm {
        //     // forget parameters
        // pub w_f: Tensor,
        // pub b_f: Tensor,
        // // input parameters
        // pub w_i: Tensor,
        // pub b_i: Tensor,
        // pub w_c: Tensor,
        // pub b_c: Tensor,
        // // output parameters
        // pub w_o: Tensor,
        // pub b_o: Tensor,

        let uniform = Uniform::new(
            -(1.0 / (hidden as f64)).sqrt(),
            (1.0 / (hidden as f64)).sqrt()
        ).unwrap();

        // forget parameters
        let w_f = Tensor::from_shape_fn(vec![hidden * 2, hidden], || {
            uniform.sample(&mut self.rng)
        });
        self.parameters.lock().unwrap().push(w_f.node.clone());

        let b_f = Tensor::from_shape_fn(vec![1, hidden], || { uniform.sample(&mut self.rng) });
        self.parameters.lock().unwrap().push(b_f.node.clone());

        // input parameters
        let w_i = Tensor::from_shape_fn(vec![hidden * 2, hidden], || {
            uniform.sample(&mut self.rng)
        });
        self.parameters.lock().unwrap().push(w_i.node.clone());

        let b_i = Tensor::from_shape_fn(vec![1, hidden], || { uniform.sample(&mut self.rng) });
        self.parameters.lock().unwrap().push(b_i.node.clone());

        let w_c = Tensor::from_shape_fn(vec![hidden * 2, hidden], || {
            uniform.sample(&mut self.rng)
        });
        self.parameters.lock().unwrap().push(w_c.node.clone());

        let b_c = Tensor::from_shape_fn(vec![1, hidden], || { uniform.sample(&mut self.rng) });
        self.parameters.lock().unwrap().push(b_c.node.clone());

        // output parameters
        let w_o = Tensor::from_shape_fn(vec![hidden * 2, hidden], || {
            uniform.sample(&mut self.rng)
        });
        self.parameters.lock().unwrap().push(w_o.node.clone());

        let b_o = Tensor::from_shape_fn(vec![1, hidden], || { uniform.sample(&mut self.rng) });
        self.parameters.lock().unwrap().push(b_o.node.clone());

        Lstm {
            w_f,
            b_c,
            w_i,
            b_i,
            w_c,
            b_f,
            b_o,
            w_o,
        }
    }
}
