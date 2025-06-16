use std::ops::{ Add, Div, Sub, Mul };

use crate::rotta_rs::{ arrayy_external::{ add_arr, divided, dot_arr }, * };

impl Add for Arrayy {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        add_arr(&self, &rhs)
    }
}

impl Sub for Arrayy {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        minus(&self, &rhs)
    }
}

impl Div for Arrayy {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        divided(&self, &rhs)
    }
}

impl Mul for Arrayy {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        mul(&self, &rhs)
    }
}

impl Arrayy {
    pub fn exp(&self) -> Arrayy {
        exp(self)
    }

    pub fn dot(&self, rhs: Arrayy) -> Arrayy {
        dot_arr(self, &rhs)
    }

    //     pub fn permute(&mut self, order: Vec<usize>) {
    //         self.update_from(permute(order, self));
    //     }

    //     pub fn slice(&mut self, slice_vector: Vec<ArrSlice>) {
    //         self.update_from(slice(&self, slice_vector));
    //     }

    //     pub fn t(&mut self){

    //     }
}
