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

// f64

impl Add<f64> for Arrayy {
    type Output = Arrayy;
    fn add(self, rhs: f64) -> Self::Output {
        add_arr(&self, &Arrayy::from_vector(vec![1], vec![rhs]))
    }
}

impl Add<f64> for &Arrayy {
    type Output = Arrayy;
    fn add(self, rhs: f64) -> Self::Output {
        add_arr(self, &Arrayy::from_vector(vec![1], vec![rhs]))
    }
}

impl Add<Arrayy> for f64 {
    type Output = Arrayy;
    fn add(self, rhs: Arrayy) -> Self::Output {
        let a = Arrayy::from_vector(vec![1], vec![self]);
        add_arr(&a, &rhs)
    }
}

impl Add<&Arrayy> for f64 {
    type Output = Arrayy;
    fn add(self, rhs: &Arrayy) -> Self::Output {
        let a = Arrayy::from_vector(vec![1], vec![self]);
        add_arr(&a, rhs)
    }
}
