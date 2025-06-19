use crate::rotta_rs::{ add, matmul, Tensor };

pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Linear {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        &matmul(x, &self.weight) + &self.bias
    }
}
