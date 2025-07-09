mod layer_norm;
pub use layer_norm::*;

use std::sync::{ Arc, Mutex };

use crate::{ Module, Tensor };

impl Module {
    pub fn layer_norm_init(&mut self, layer_shape: &[usize]) -> LayerNorm {
        let mut parameters = self.parameters.lock().unwrap();

        let shape = layer_shape.to_vec();
        let gamma = Tensor::from_element(shape.clone(), 1.0);
        parameters.push(gamma.node.clone());

        let beta = Tensor::from_element(shape.clone(), 0.0);
        parameters.push(beta.node.clone());

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
