mod base_operation;
pub use base_operation::*;

use std::ops::{ Add, Div, Sub, Mul };

use crate::rotta_rs::{ arrayy_external::{ add_arr, divided, dot_arr }, * };

impl Arrayy {
    pub fn sum(&self) -> f64 {
        self.value.iter().sum::<f64>()
    }

    pub fn exp(&self) -> Arrayy {
        exp(self)
    }

    pub fn dot(&self, rhs: &Arrayy) -> Arrayy {
        dot_arr(self, rhs)
    }

    pub fn matmul(&self, rhs: &Arrayy) -> Arrayy {
        matmul_nd(self, rhs)
    }

    pub fn permute(&self, order: Vec<usize>) -> Arrayy {
        permute(order, self)
    }

    //     pub fn slice(&mut self, slice_vector: Vec<ArrSlice>) {
    //         self.update_from(slice(&self, slice_vector));
    //     }

    pub fn t(&self) -> Arrayy {
        transpose(&self, (-1, -2))
    }

    pub fn powi(&self, n: i32) -> Arrayy {
        powi_arr(self, n)
    }

    pub fn powf(&self, n: f64) -> Arrayy {
        powf_arr(self, n)
    }

    pub fn unsqueeze(&self) -> Arrayy {
        let shape = self.shape.clone()[1..].to_vec();

        Arrayy::from_vector(shape, self.value.clone())
    }
}
