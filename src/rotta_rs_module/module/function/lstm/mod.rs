mod lstm;
pub use lstm::*;
pub use rand::distr::Uniform;
use rand_distr::Distribution;

use crate::{ Module, Tensor };

impl Module {
    pub fn lstm_init(&mut self, hidden: usize) -> Lstm {
        let uniform = Uniform::new(
            -(1.0 / (hidden as f64)).sqrt(),
            (1.0 / (hidden as f64)).sqrt()
        ).unwrap();

        // forget parameters
        let w_f = Tensor::from_shape_fn(vec![hidden * 2, hidden], || {
            uniform.sample(&mut self.rng)
        });
        w_f.set_auto_zero_grad(false);
        self.parameters.lock().unwrap().push(w_f.shared_tensor());

        let b_f = Tensor::from_shape_fn(vec![1, hidden], || { uniform.sample(&mut self.rng) });
        b_f.set_auto_zero_grad(false);
        self.parameters.lock().unwrap().push(b_f.shared_tensor());

        // input parameters
        let w_i = Tensor::from_shape_fn(vec![hidden * 2, hidden], || {
            uniform.sample(&mut self.rng)
        });
        w_i.set_auto_zero_grad(false);
        self.parameters.lock().unwrap().push(w_i.shared_tensor());

        let b_i = Tensor::from_shape_fn(vec![1, hidden], || { uniform.sample(&mut self.rng) });
        b_i.set_auto_zero_grad(false);
        self.parameters.lock().unwrap().push(b_i.shared_tensor());

        let w_c = Tensor::from_shape_fn(vec![hidden * 2, hidden], || {
            uniform.sample(&mut self.rng)
        });
        w_c.set_auto_zero_grad(false);
        self.parameters.lock().unwrap().push(w_c.shared_tensor());

        let b_c = Tensor::from_shape_fn(vec![1, hidden], || { uniform.sample(&mut self.rng) });
        b_c.set_auto_zero_grad(false);
        self.parameters.lock().unwrap().push(b_c.shared_tensor());

        // output parameters
        let w_o = Tensor::from_shape_fn(vec![hidden * 2, hidden], || {
            uniform.sample(&mut self.rng)
        });
        w_o.set_auto_zero_grad(false);
        self.parameters.lock().unwrap().push(w_o.shared_tensor());

        let b_o = Tensor::from_shape_fn(vec![1, hidden], || { uniform.sample(&mut self.rng) });
        b_o.set_auto_zero_grad(false);
        self.parameters.lock().unwrap().push(b_o.shared_tensor());

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
