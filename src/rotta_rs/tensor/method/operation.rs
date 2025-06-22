use crate::rotta_rs::{ abs, powi, sum, Tensor };

impl Tensor {
    pub fn powi(&self, n: i32) -> Tensor {
        powi(self, n)
    }

    pub fn sum(&self) -> Tensor {
        sum(self)
    }

    pub fn len(&self) -> usize {
        self.node.lock().unwrap().value.len()
    }

    pub fn abs(&self) -> Tensor {
        abs(self)
    }
}
