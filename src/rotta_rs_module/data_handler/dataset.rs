use crate::Tensor;

pub trait Dataset {
    fn get(&self, idx: usize) -> (Tensor, Tensor);
    fn len(&self) -> usize;
}
