use std::sync::{ Arc, Mutex };

use crate::{ Tensor, TrainEvalHandler };

#[derive(Clone)]
pub struct BatchNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub eps: f64,
    pub alpha: f64,
    pub r_mean: Tensor,
    pub r_variant: Tensor,
    pub eval: Arc<Mutex<bool>>,
}

impl BatchNorm {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        if shape.len() <= 1 {
            panic!("error, BatchNorm input must have minimum have 2dimension");
        }

        if !self.eval_status() {
            // [N,C,W,H]
            let mut axis = vec![];
            shape
                .into_iter()
                .enumerate()
                .for_each(|(i, _)| {
                    if i != 1 {
                        axis.push(i as i32);
                    }
                });

            let mean = x.mean_axis_keep_dim(&axis);
            let variant = (x - &mean).powi(2).mean_axis_keep_dim(&axis);

            let eps = self.eps;
            let norm = &(x - &mean) / &(&variant + eps).powf(0.5);

            // update running value
            // r_mean = r_mean * (1 - alpha) + mean * alpha
            let r_mean =
                &(&self.r_mean.value() * (1.0 - self.alpha)) + &(&mean.value() * self.alpha);
            self.r_mean.update_value(r_mean);

            // r_variant = r_variant * (1 - alpha) + variant * alpha
            let r_variant =
                &(&self.r_variant.value() * (1.0 - self.alpha)) + &(&variant.value() * self.alpha);
            self.r_variant.update_value(r_variant);

            // output
            &(&self.gamma * &norm) + &self.beta
            // norm
        } else {
            let eps = 1e-8;
            self.r_mean.set_requires_grad(false);
            self.r_variant.set_requires_grad(false);

            let norm = &(x - &self.r_mean) / &(&self.r_variant + eps).powf(0.5);
            &(&self.gamma * &norm) + &self.beta
        }
    }

    pub fn eval_status(&self) -> bool {
        *self.eval.lock().unwrap()
    }
}

impl TrainEvalHandler for BatchNorm {
    fn eval(&mut self) {
        *self.eval.lock().unwrap() = true;
    }

    fn train(&mut self) {
        *self.eval.lock().unwrap() = false;
    }
}
