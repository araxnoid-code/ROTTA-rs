use std::ops::Sub;

use crate::rotta_rs_module::{ arrayy::{ Arrayy, * } };

impl Sub for Arrayy {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        minus(&self, &rhs)
    }
}

impl Sub<&Arrayy> for &Arrayy {
    type Output = Arrayy;
    fn sub(self, rhs: &Arrayy) -> Self::Output {
        minus(self, rhs)
    }
}

impl Sub<&Arrayy> for Arrayy {
    type Output = Arrayy;
    fn sub(self, rhs: &Arrayy) -> Self::Output {
        minus(&self, rhs)
    }
}

impl Sub<Arrayy> for &Arrayy {
    type Output = Arrayy;
    fn sub(self, rhs: Arrayy) -> Self::Output {
        minus(self, &rhs)
    }
}

// f32

impl Sub<f32> for Arrayy {
    type Output = Arrayy;
    fn sub(self, rhs: f32) -> Self::Output {
        minus(&self, &Arrayy::from_vector(vec![1], vec![rhs]))
    }
}

impl Sub<f32> for &Arrayy {
    type Output = Arrayy;
    fn sub(self, rhs: f32) -> Self::Output {
        minus(self, &Arrayy::from_vector(vec![1], vec![rhs]))
    }
}

impl Sub<Arrayy> for f32 {
    type Output = Arrayy;
    fn sub(self, rhs: Arrayy) -> Self::Output {
        let a = Arrayy::from_vector(vec![1], vec![self]);
        minus(&a, &rhs)
    }
}

impl Sub<&Arrayy> for f32 {
    type Output = Arrayy;
    fn sub(self, rhs: &Arrayy) -> Self::Output {
        let a = Arrayy::from_vector(vec![1], vec![self]);
        minus(&a, rhs)
    }
}
