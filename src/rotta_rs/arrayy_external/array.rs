use std::{ fmt::Display, vec };

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
        let length = shape.as_slice().multiple_sum();

        if length != vector.len() {
            panic!("shape and length of vector not same");
        }

        let arr = Arrayy {
            value: vector,
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
            value: vec![1.0; length],
        };
        arr
    }

    pub fn arrayy_from_shape_fn<F: FnMut() -> f64>(shape: Vec<usize>, mut f: F) -> Arrayy {
        let mut vector = vec![];
        for _ in 0..shape.as_slice().multiple_sum() {
            vector.push(f());
        }

        Arrayy::from_vector(shape, vector)
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

    // method
    pub fn map<F: FnMut(&f64) -> f64>(&self, f: F) -> Arrayy {
        let vector = self.value.iter().map(f).collect::<Vec<f64>>();

        Arrayy::from_vector(self.shape.clone(), vector)
    }

    // update
    pub fn update_from(&mut self, arr: Arrayy) {
        self.value = arr.value;
        self.shape = arr.shape;
    }
}
