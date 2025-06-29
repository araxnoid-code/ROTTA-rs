use std::vec;

use crate::rotta_rs_module::{ arrayy::{ negative_indexing, MultipleSum, RecFlatten }, Tensor };

#[derive(Clone, Debug)]
pub struct Arrayy {
    pub value: Vec<f64>,
    pub shape: Vec<usize>,
}

impl Arrayy {
    // create

    pub fn new<T: RecFlatten>(arr: T) -> Arrayy {
        let (shape, vector) = arr.rec_flatten();
        Arrayy::from_vector(shape, vector)
    }

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
        let len = shape.multiple_sum();
        let mut vector = Vec::with_capacity(len);
        for _ in 0..len {
            vector.push(f());
        }

        Arrayy::from_vector(shape, vector)
    }

    pub fn arrayy_from_element(shape: Vec<usize>, element: f64) -> Arrayy {
        let len = shape.multiple_sum();
        let mut vector = Vec::with_capacity(len);
        for _ in 0..len {
            vector.push(element);
        }

        Arrayy::from_vector(shape, vector)
    }

    // get
    pub fn index(&self, index: Vec<i32>) -> Arrayy {
        let shape = self.shape.clone();
        let index = negative_indexing(&self.shape, index).unwrap();

        let mut out = 0;
        let mut count = 1;
        let slicing = (&self.shape[index.len()..]).multiple_sum();
        let mut new_shape = self.shape[index.len()..].to_vec();
        if new_shape.is_empty() {
            new_shape.push(1);
        }

        for index in index {
            let multiple = (&shape[count..]).multiple_sum();
            let pointing = index * multiple;
            out += pointing;

            count += 1;
        }

        let arrayy = Arrayy::from_vector(new_shape, self.value[out..out + slicing].to_vec());

        arrayy
    }

    pub fn index_mut(&mut self, index: Vec<i32>, value: Arrayy) {
        let shape = self.shape.clone();
        let index = negative_indexing(&self.shape, index).unwrap();
        let slicing = (&self.shape[index.len()..]).multiple_sum();

        let mut out = 0;
        let mut count = 1;
        for index in index {
            let multiple = (&shape[count..]).multiple_sum();
            let pointing = index * multiple;
            out += pointing;

            count += 1;
        }

        self.value[out..out + slicing].copy_from_slice(&value.value[..]);
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
