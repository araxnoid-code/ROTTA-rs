use std::ops::Div;

use crate::rotta_rs_module::{ arrayy::{ Arrayy, * } };

impl Div for Arrayy {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        divided_arr(&self, &rhs)
    }
}

impl Div<&Arrayy> for &Arrayy {
    type Output = Arrayy;
    fn div(self, rhs: &Arrayy) -> Self::Output {
        divided_arr(self, rhs)
    }
}

impl Div<&Arrayy> for Arrayy {
    type Output = Arrayy;
    fn div(self, rhs: &Arrayy) -> Self::Output {
        divided_arr(&self, rhs)
    }
}

impl Div<Arrayy> for &Arrayy {
    type Output = Arrayy;
    fn div(self, rhs: Arrayy) -> Self::Output {
        divided_arr(self, &rhs)
    }
}

// f32

impl Div<f32> for Arrayy {
    type Output = Arrayy;
    fn div(self, rhs: f32) -> Self::Output {
        divided_arr(&self, &Arrayy::from_vector(vec![1], vec![rhs]))
    }
}

impl Div<f32> for &Arrayy {
    type Output = Arrayy;
    fn div(self, rhs: f32) -> Self::Output {
        divided_arr(self, &Arrayy::from_vector(vec![1], vec![rhs]))
    }
}

impl Div<Arrayy> for f32 {
    type Output = Arrayy;
    fn div(self, rhs: Arrayy) -> Self::Output {
        let a = Arrayy::from_vector(vec![1], vec![self]);
        divided_arr(&a, &rhs)
    }
}

impl Div<&Arrayy> for f32 {
    type Output = Arrayy;
    fn div(self, rhs: &Arrayy) -> Self::Output {
        let a = Arrayy::from_vector(vec![1], vec![self]);
        divided_arr(&a, rhs)
    }
}
