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
        gamma.set_auto_zero_grad(false);
        parameters.push(gamma.shared_tensor());

        let beta = Tensor::from_element(shape.clone(), 0.0);
        beta.set_auto_zero_grad(false);
        parameters.push(beta.shared_tensor());

        let r_mean = Tensor::from_element(shape.clone(), 0.0);
        r_mean.set_auto_zero_grad(false);
        r_mean.set_requires_grad(false);

        let r_variant = Tensor::from_element(shape.clone(), 0.0);
        r_variant.set_auto_zero_grad(false);
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
