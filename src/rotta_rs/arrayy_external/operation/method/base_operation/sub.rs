use std::ops::Sub;

use crate::rotta_rs::{ minus, Arrayy };

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
