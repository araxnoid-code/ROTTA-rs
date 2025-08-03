mod layer_norm;
pub use layer_norm::*;

use std::sync::{ Arc, Mutex };

use crate::{ Module, Tensor };

impl Module {
    pub fn layer_norm_init(&mut self, layer_shape: &[usize]) -> LayerNorm {
        let mut parameters = self.parameters.lock().unwrap();

        let shape = layer_shape.to_vec();
        let gamma = Tensor::from_element(shape.clone(), 1.0);
        gamma.set_auto_zero_grad(false);
        parameters.push(gamma.shared_tensor());

        let beta = Tensor::from_element(shape.clone(), 0.0);
        beta.set_auto_zero_grad(false);
        parameters.push(beta.shared_tensor());

        let eval_status = Arc::new(Mutex::new(false));
        self.eval_handlers.push(eval_status.clone());
        let batch_norm = LayerNorm {
            beta: beta,
            gamma: gamma,
            eps: 1e-8,
            alpha: 0.1,
            eval: eval_status,
            learnable: true,
        };

        batch_norm
    }
}
