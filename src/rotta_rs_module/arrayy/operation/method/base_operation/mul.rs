use std::ops::Mul;

use crate::rotta_rs_module::{ arrayy::{ Arrayy, * } };

impl Mul for Arrayy {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        mul_arr(&self, &rhs)
    }
}

impl Mul<&Arrayy> for &Arrayy {
    type Output = Arrayy;
    fn mul(self, rhs: &Arrayy) -> Self::Output {
        mul_arr(self, rhs)
    }
}

impl Mul<Arrayy> for &Arrayy {
    type Output = Arrayy;
    fn mul(self, rhs: Arrayy) -> Self::Output {
        mul_arr(self, &rhs)
    }
}

impl Mul<&Arrayy> for Arrayy {
    type Output = Arrayy;
    fn mul(self, rhs: &Arrayy) -> Self::Output {
        mul_arr(&self, rhs)
    }
}

// f64

impl Mul<f64> for Arrayy {
    type Output = Arrayy;
    fn mul(self, rhs: f64) -> Self::Output {
        mul_arr(&self, &Arrayy::from_vector(vec![1], vec![rhs]))
    }
}

impl Mul<f64> for &Arrayy {
    type Output = Arrayy;
    fn mul(self, rhs: f64) -> Self::Output {
        mul_arr(self, &Arrayy::from_vector(vec![1], vec![rhs]))
    }
}

impl Mul<Arrayy> for f64 {
    type Output = Arrayy;
    fn mul(self, rhs: Arrayy) -> Self::Output {
        let a = Arrayy::from_vector(vec![1], vec![self]);
        mul_arr(&a, &rhs)
    }
}

impl Mul<&Arrayy> for f64 {
    type Output = Arrayy;
    fn mul(self, rhs: &Arrayy) -> Self::Output {
        let a = Arrayy::from_vector(vec![1], vec![self]);
        mul_arr(&a, rhs)
    }
}
