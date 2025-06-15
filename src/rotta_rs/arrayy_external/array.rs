use std::fmt::Display;

use crate::rotta_rs::{ MultipleSum, RecFlatten };

#[derive(Clone)]
pub struct Arrayy {
    pub value: Vec<f64>,
    pub shape: Vec<usize>,
}

impl Arrayy {
    pub fn from_vector(shape: Vec<usize>, vector: Vec<f64>) -> Arrayy {
        let flatten = vector.rec_flatten();

        let arr = Arrayy {
            value: flatten,
            shape: shape,
        };

        arr
    }

    pub fn index(&self, index: &[usize]) -> f64 {
        let shape = self.shape.clone();

        let mut out = 0;
        let mut count = 1;
        for index in index {
            let multiple = (&shape[count..]).multiple_sum();
            let pointing = index * multiple;
            out += pointing;

            count += 1;
        }

        self.value[out]
    }

    pub fn index_mut(&mut self, index: &[usize], value: f64) {
        let shape = self.shape.clone();

        let mut out = 0;
        let mut count = 1;
        for index in index {
            let multiple = (&shape[count..]).multiple_sum();
            let pointing = index * multiple;
            out += pointing;

            count += 1;
        }

        self.value[out] = value;
    }

    pub fn update_from(&mut self, arr: Arrayy) {
        self.value = arr.value;
        self.shape = arr.shape;
    }
}
