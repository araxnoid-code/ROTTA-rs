use std::ops::{ Add, Div };

use crate::rotta_rs::{ add, divided, Tensor };

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Self::Output {
        add(self, rhs)
    }
}

impl Add<f64> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: f64) -> Self::Output {
        let rhs = Tensor::from_vector(vec![1], vec![rhs]);
        add(self, &rhs)
    }
}

impl Add<&Tensor> for f64 {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Self::Output {
        let float = Tensor::from_vector(vec![1], vec![self]);
        add(&float, rhs)
    }
}

//

impl Div<&Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Self::Output {
        divided(self, rhs)
    }
}
