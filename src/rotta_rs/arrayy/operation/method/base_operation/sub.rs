use std::ops::Sub;

use crate::rotta_rs::{ arrayy::{ add_arr, Arrayy, * } };

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

// f64

impl Sub<f64> for Arrayy {
    type Output = Arrayy;
    fn sub(self, rhs: f64) -> Self::Output {
        minus(&self, &Arrayy::from_vector(vec![1], vec![rhs]))
    }
}

impl Sub<f64> for &Arrayy {
    type Output = Arrayy;
    fn sub(self, rhs: f64) -> Self::Output {
        minus(self, &Arrayy::from_vector(vec![1], vec![rhs]))
    }
}

impl Sub<Arrayy> for f64 {
    type Output = Arrayy;
    fn sub(self, rhs: Arrayy) -> Self::Output {
        let a = Arrayy::from_vector(vec![1], vec![self]);
        minus(&a, &rhs)
    }
}

impl Sub<&Arrayy> for f64 {
    type Output = Arrayy;
    fn sub(self, rhs: &Arrayy) -> Self::Output {
        let a = Arrayy::from_vector(vec![1], vec![self]);
        minus(&a, rhs)
    }
}
