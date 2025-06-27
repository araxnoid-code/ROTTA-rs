use crate::rotta_rs::{ add, matmul, Tensor, TrainEvalHandler };

pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Linear {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        &matmul(x, &self.weight) + &self.bias
    }
}

// training evaluation handler
impl TrainEvalHandler for Linear {
    fn train(&mut self) {}
    fn eval(&mut self) {}
}
