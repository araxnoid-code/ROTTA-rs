use std::ops::Add;

use crate::rotta_rs_module::{ arrayy::{ add_arr, Arrayy } };

impl Add for Arrayy {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        add_arr(&self, &rhs)
    }
}

impl Add<&Arrayy> for &Arrayy {
    type Output = Arrayy;
    fn add(self, rhs: &Arrayy) -> Self::Output {
        add_arr(self, rhs)
    }
}

impl Add<Arrayy> for &Arrayy {
    type Output = Arrayy;
    fn add(self, rhs: Arrayy) -> Self::Output {
        add_arr(self, &rhs)
    }
}

impl Add<&Arrayy> for Arrayy {
    type Output = Arrayy;
    fn add(self, rhs: &Arrayy) -> Self::Output {
        add_arr(&self, rhs)
    }
}

// f32

impl Add<f32> for Arrayy {
    type Output = Arrayy;
    fn add(self, rhs: f32) -> Self::Output {
        add_arr(&self, &Arrayy::from_vector(vec![1], vec![rhs]))
    }
}

impl Add<f32> for &Arrayy {
    type Output = Arrayy;
    fn add(self, rhs: f32) -> Self::Output {
        add_arr(self, &Arrayy::from_vector(vec![1], vec![rhs]))
    }
}

impl Add<Arrayy> for f32 {
    type Output = Arrayy;
    fn add(self, rhs: Arrayy) -> Self::Output {
        let a = Arrayy::from_vector(vec![1], vec![self]);
        add_arr(&a, &rhs)
    }
}

impl Add<&Arrayy> for f32 {
    type Output = Arrayy;
    fn add(self, rhs: &Arrayy) -> Self::Output {
        let a = Arrayy::from_vector(vec![1], vec![self]);
        add_arr(&a, rhs)
    }
}
