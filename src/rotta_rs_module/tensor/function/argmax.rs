use crate::Tensor;

pub fn argmax(tensor: &Tensor, dim: i32) -> Tensor {
    Tensor::from_arrayy(tensor.value().argmax(dim))
}
// argmax has no derivative
