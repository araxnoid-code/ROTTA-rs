use std::fmt::Display;

use uuid::Uuid;

use crate::rotta_rs::{ BackwardLabel, MultipleSum, NodeType, RecFlatten };

#[derive(Clone, Debug)]
pub struct Arrayy {
    pub value: Vec<f64>,
    pub shape: Vec<usize>,
}

impl Arrayy {
    // create
    pub fn from_vector(shape: Vec<usize>, vector: Vec<f64>) -> Arrayy {
        let flatten = vector.rec_flatten();

        let arr = Arrayy {
            value: flatten.clone(),
            shape: shape,
        };

        arr
    }

    pub fn zeros(shape: Vec<usize>) -> Arrayy {
        let length = shape.as_slice().multiple_sum();

        let arr = Arrayy {
            shape,
            value: vec![0.0; length],
        };
        arr
    }

    pub fn ones(shape: Vec<usize>) -> Arrayy {
        let length = shape.as_slice().multiple_sum();

        let arr = Arrayy {
            shape,
            value: vec![0.0; length],
        };
        arr
    }

    // get
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

    // update
    pub fn update_from(&mut self, arr: Arrayy) {
        self.value = arr.value;
        self.shape = arr.shape;
    }
}
