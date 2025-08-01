use crate::Tensor;

pub fn argmin(tensor: &Tensor, dim: i32) -> Tensor {
    Tensor::from_arrayy(tensor.value().argmin(dim))
}
// argmin has no derivative
