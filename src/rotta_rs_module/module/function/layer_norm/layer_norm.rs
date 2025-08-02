use std::sync::{ Arc, Mutex };

use crate::{ Tensor, TrainEvalHandler };

#[derive(Clone)]
pub struct LayerNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub eps: f64,
    pub alpha: f64,
    pub eval: Arc<Mutex<bool>>,
    pub learnable: bool,
}

impl LayerNorm {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();

        if shape[1..] != self.gamma.shape() {
            println!(
                "LAYER NORM ERROR: input shape does not match gamma, input shape:{:?} gamma shape:{:?}",
                shape,
                self.gamma.shape()
            );
        }

        let mut axis = vec![];
        shape
            .into_iter()
            .enumerate()
            .for_each(|(i, _)| {
                if i != 0 {
                    axis.push(i as i32);
                }
            });

        let mean = x.mean_axis_keep_dim(&axis);
        let variant = (x - &mean).powi(2).mean_axis_keep_dim(&axis);

        let eps = self.eps;
        let norm = &(x - &mean) / &(&variant + eps).powf(0.5);

        // output
        if self.learnable {
            &(&self.gamma * &norm) + &self.beta
        } else {
            norm
        }
    }

    pub fn disable_learnable(&mut self) {
        self.learnable = false;
    }

    pub fn enable_learnable(&mut self) {
        self.learnable = true;
    }

    pub fn eval_status(&self) -> bool {
        *self.eval.lock().unwrap()
    }
}

impl TrainEvalHandler for LayerNorm {
    fn eval(&mut self) {
        *self.eval.lock().unwrap() = true;
    }

    fn train(&mut self) {
        *self.eval.lock().unwrap() = false;
    }
}
