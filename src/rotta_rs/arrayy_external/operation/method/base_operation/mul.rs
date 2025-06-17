use std::ops::Mul;

use crate::rotta_rs::{ mul, Arrayy };

impl Mul for Arrayy {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        mul(&self, &rhs)
    }
}

impl Mul<&Arrayy> for &Arrayy {
    type Output = Arrayy;
    fn mul(self, rhs: &Arrayy) -> Self::Output {
        mul(self, rhs)
    }
}

impl Mul<Arrayy> for &Arrayy {
    type Output = Arrayy;
    fn mul(self, rhs: Arrayy) -> Self::Output {
        mul(self, &rhs)
    }
}

impl Mul<&Arrayy> for Arrayy {
    type Output = Arrayy;
    fn mul(self, rhs: &Arrayy) -> Self::Output {
        mul(&self, rhs)
    }
}
