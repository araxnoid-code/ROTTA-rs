use std::ops::Div;

use crate::rotta_rs::{ divided_arr, Arrayy };

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
