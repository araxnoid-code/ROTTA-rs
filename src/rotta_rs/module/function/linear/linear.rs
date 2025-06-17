use crate::rotta_rs::{ add, matmul, NdArray, Tensor };

pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Linear {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mul = &matmul(x, &self.weight);
        // println!("{}", mul);

        let output = add(&self.bias, mul);

        output
    }
}
