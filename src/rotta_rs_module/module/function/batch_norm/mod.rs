mod batch_norm;
use std::sync::{ Arc, Mutex };

pub use batch_norm::*;

use crate::{ Module, Tensor };

impl Module {
    pub fn batch_norm_init(
        &mut self,
        channel_features: usize,
        input_dimension: usize
    ) -> BatchNorm {
        let mut parameters = self.parameters.lock().unwrap();

        let mut shape = vec![1;input_dimension];
        shape[1] = channel_features;
        let gamma = Tensor::from_element(shape.clone(), 1.0);
        parameters.push(gamma.node.clone());

        let beta = Tensor::from_element(shape.clone(), 0.0);
        parameters.push(beta.node.clone());

        let r_mean = Tensor::from_element(shape.clone(), 0.0);
        r_mean.set_requires_grad(false);

        let r_variant = Tensor::from_element(shape.clone(), 0.0);
        r_variant.set_requires_grad(false);

        let eval_status = Arc::new(Mutex::new(false));
        self.eval_handlers.push(eval_status.clone());
        let batch_norm = BatchNorm {
            beta: beta,
            gamma: gamma,
            eps: 1e-8,
            alpha: 0.1,
            r_mean,
            r_variant,
            eval: eval_status,
        };

        batch_norm
    }
}
