use std::ops::{ Add, Div, Mul };

use crate::rotta_rs::{ add, divided, mul, Tensor };

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

impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Self::Output {
        mul(self, rhs)
    }
}

impl Mul<f64> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f64) -> Self::Output {
        let rhs = Tensor::from_vector(vec![1], vec![rhs]);
        mul(self, &rhs)
    }
}

impl Mul<&Tensor> for f64 {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Self::Output {
        let float = Tensor::from_vector(vec![1], vec![self]);
        mul(&float, rhs)
    }
}

//

impl Div<&Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Self::Output {
        divided(self, rhs)
    }
}

impl Div<f64> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: f64) -> Self::Output {
        let rhs = Tensor::from_vector(vec![1], vec![rhs]);
        divided(self, &rhs)
    }
}

impl Div<&Tensor> for f64 {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Self::Output {
        let float = Tensor::from_vector(vec![1], vec![self]);
        divided(&float, rhs)
    }
}
