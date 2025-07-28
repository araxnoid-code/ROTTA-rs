use crate::rotta_rs_module::Tensor;

#[derive(Clone)]
pub struct MSE {}

impl MSE {
    pub fn init() -> Self {
        Self {}
    }

    pub fn forward(&self, prediction: &Tensor, label: &Tensor) -> Tensor {
        (1.0 / (prediction.len() as f64)) * &(label - prediction).powi(2).sum()
    }
}
