use std::ops::Add;

use crate::rotta_rs::{ add_arr, Arrayy };

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
