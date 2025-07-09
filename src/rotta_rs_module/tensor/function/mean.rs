use crate::Tensor;

pub fn mean(x: &Tensor) -> Tensor {
    &x.sum() / (x.len() as f64)
}
