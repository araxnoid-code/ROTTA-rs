use crate::rotta_rs_module::{ matmul, Tensor };

#[derive(Clone)]
pub struct Linear {
    pub input: usize,
    pub output: usize,
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Linear {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        if shape.len() != 2 {
            panic!(
                "Linear Erros: Linear function only able for 2d tensor, input shape:{:?}",
                shape
            );
        } else if shape[1] != self.input {
            panic!(
                "Linear Error: Linear function only able for {} features, input shape:{:?}",
                self.input,
                shape
            );
        }
        &matmul(x, &self.weight) + &self.bias
    }
}
