use crate::rotta_rs_module::Tensor;

pub struct MAE {}

impl MAE {
    pub fn init() -> Self {
        Self {}
    }

    pub fn forward(&self, prediction: &Tensor, label: &Tensor) -> Tensor {
        // 1/n sum(abs(label - prediction))
        (1.0 / (prediction.len() as f64)) * &(label - prediction).abs().sum()
    }
}
