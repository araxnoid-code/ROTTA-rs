use crate::Tensor;

pub fn flatten(x: &Tensor) -> Tensor {
    x.reshape(vec![-1])
}
